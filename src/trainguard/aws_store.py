"""S3 and DynamoDB adapter for the remote checkpoint protocol.

Immutable generation objects live in S3 and are only created with
``If-None-Match: *``. The mutable HEAD and AUTHORITY records live in one DynamoDB
table with a per-key revision counter: an S3 ETag is a content digest, so it
returns to an old value after A->B->A and cannot serve as a CAS token. Mutable
records are never deleted, so a revision token is never reused.

boto3 is imported only by ``connect``; the adapter itself takes client objects.
Passing the unit tests here does not validate a real bucket, table, permission
model, or service failure behavior.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from trainguard.remote_protocol import (
    PreconditionFailed,
    RemoteProtocolError,
    ResponseLost,
    StoredObject,
)

_MUTABLE_KEY = re.compile(r"runs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/(HEAD|AUTHORITY)\Z")
_GENERATION_PREFIX = re.compile(r"runs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/generations/")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_TABLE = re.compile(r"[A-Za-z0-9_.-]{3,255}\Z")
_REGION = re.compile(r"[a-z]{2}(-[a-z]+)+-[0-9]\Z")
_PREFIX_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REVISION = re.compile(r"revision-([1-9][0-9]{0,18})\Z")
# DynamoDB items are capped at 400 KB including attribute names and the key.
MAX_MUTABLE_BYTES = 256 * 1024
CONFLICT_RETRIES = 3
_TRANSIENT_CODES = {
    "InternalError", "InternalServerError", "ServiceUnavailable", "SlowDown",
    "RequestTimeout", "ThrottlingException", "ProvisionedThroughputExceededException",
    "RequestLimitExceeded", "TransactionConflictException",
}


@dataclass(frozen=True)
class AwsLocation:
    bucket: str
    prefix: str
    table: str
    region: str

    @property
    def uri(self) -> str:
        return f"aws://{self.bucket}/{self.prefix}?region={self.region}&table={self.table}"


def is_aws_location(value: object) -> bool:
    return isinstance(value, str) and value.startswith("aws://")


def parse_location(value: str) -> AwsLocation:
    """Accept only the canonical form so a run's store identity is one string."""
    parts = urlsplit(value)
    try:
        query = parse_qs(parts.query, strict_parsing=True)
    except ValueError as exc:
        raise ValueError("AWS reference location query is invalid") from exc
    prefix = parts.path.removeprefix("/")
    if (
        parts.scheme != "aws" or parts.fragment or parts.username or parts.password
        or parts.port is not None or set(query) != {"region", "table"}
        or any(len(items) != 1 for items in query.values())
        or not prefix or any(_PREFIX_PART.fullmatch(part) is None for part in prefix.split("/"))
    ):
        raise ValueError(
            "AWS reference location must be aws://<bucket>/<prefix>?region=<region>&table=<table>"
        )
    location = AwsLocation(parts.hostname or "", prefix, query["table"][0], query["region"][0])
    if (
        _BUCKET.fullmatch(location.bucket) is None
        or ".." in location.bucket
        or _TABLE.fullmatch(location.table) is None
        or _REGION.fullmatch(location.region) is None
        or location.uri != value
    ):
        raise ValueError(f"AWS reference location is not canonical; expected {location.uri}")
    return location


def _error(exc: BaseException) -> tuple[str | None, int | None]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None, None
    error = response.get("Error") or {}
    metadata = response.get("ResponseMetadata") or {}
    return error.get("Code"), metadata.get("HTTPStatusCode")


def _default_transport_errors() -> tuple[type[BaseException], ...]:
    try:
        from botocore.exceptions import ConnectionError as BotoConnectionError
        from botocore.exceptions import HTTPClientError
    except ImportError:
        return (ConnectionError, TimeoutError)
    return (BotoConnectionError, HTTPClientError, ConnectionError, TimeoutError)


class AwsObjectStore:
    """ObjectStore over S3 (create-only generations) and DynamoDB (mutable records)."""

    def __init__(
        self, s3, dynamodb, location: AwsLocation, *, read_only: bool = False,
        transport_errors: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self.s3 = s3
        self.dynamodb = dynamodb
        self.aws = location
        self.location = location.uri
        self.read_only = read_only
        self._transport_errors = (
            transport_errors if transport_errors is not None else _default_transport_errors()
        )

    def _full(self, key: str) -> str:
        if not isinstance(key, str) or not key or key.startswith("/") or "\x00" in key:
            raise ValueError("object key is invalid")
        return f"{self.aws.prefix}/{key}"

    def _write(self, operation):
        """Map service outcomes onto the protocol's three write results."""
        for attempt in range(CONFLICT_RETRIES + 1):
            try:
                return operation()
            except Exception as exc:
                code, status = _error(exc)
                if code in {"PreconditionFailed", "ConditionalCheckFailedException"}:
                    raise PreconditionFailed(f"conditional write rejected: {code}") from exc
                if code == "ConditionalRequestConflict":
                    # S3 409: a concurrent operation won; the write did not apply.
                    if attempt < CONFLICT_RETRIES:
                        continue
                    raise RemoteProtocolError("S3 conditional write kept conflicting") from exc
                if (
                    code in _TRANSIENT_CODES
                    or (status is not None and status >= 500)
                    or isinstance(exc, self._transport_errors)
                ):
                    raise ResponseLost(f"write outcome is unknown: {code or type(exc).__name__}") from exc
                raise
        raise AssertionError("unreachable")

    def _item(self, key: str) -> dict | None:
        response = self.dynamodb.get_item(
            TableName=self.aws.table, Key={"pk": {"S": self._full(key)}}, ConsistentRead=True,
        )
        return response.get("Item")

    def get(self, key: str) -> StoredObject | None:
        if _MUTABLE_KEY.fullmatch(key):
            item = self._item(key)
            if item is None:
                return None
            return StoredObject(bytes(item["data"]["B"]), f"revision-{int(item['revision']['N'])}")
        try:
            response = self.s3.get_object(Bucket=self.aws.bucket, Key=self._full(key))
        except Exception as exc:
            code, status = _error(exc)
            if code in {"NoSuchKey", "404"} or status == 404:
                return None
            raise
        return StoredObject(response["Body"].read(), response["ETag"])

    def object_size(self, key: str) -> int | None:
        if _MUTABLE_KEY.fullmatch(key):
            item = self._item(key)
            return None if item is None else len(item["data"]["B"])
        try:
            response = self.s3.head_object(Bucket=self.aws.bucket, Key=self._full(key))
        except Exception as exc:
            code, status = _error(exc)
            if code in {"NoSuchKey", "NotFound", "404"} or status == 404:
                return None
            raise
        return int(response["ContentLength"])

    def list(self, prefix: str) -> tuple[str, ...]:
        # ponytail: only generation prefixes are listable; mutable records are not in S3.
        if _GENERATION_PREFIX.match(prefix) is None:
            raise ValueError("only generation prefixes can be listed")
        full_prefix = self._full(prefix)
        keys: list[str] = []
        request = {"Bucket": self.aws.bucket, "Prefix": full_prefix}
        while True:
            response = self.s3.list_objects_v2(**request)
            keys.extend(item["Key"] for item in response.get("Contents", ()))
            if not response.get("IsTruncated"):
                break
            request["ContinuationToken"] = response["NextContinuationToken"]
        start = len(self.aws.prefix) + 1
        return tuple(sorted(key[start:] for key in keys))

    def put(
        self, key: str, data: bytes, *, if_none_match: bool = False, if_match: str | None = None,
    ) -> StoredObject:
        if self.read_only:
            raise ValueError("AWS reference store is read only")
        if if_none_match == (if_match is not None):
            raise ValueError("put requires exactly one conditional precondition")
        payload = bytes(data)
        full = self._full(key)
        if _MUTABLE_KEY.fullmatch(key):
            if len(payload) > MAX_MUTABLE_BYTES:
                raise ValueError("mutable record exceeds the DynamoDB item budget")
            if if_none_match:
                revision = 1
                condition: dict = {"ConditionExpression": "attribute_not_exists(pk)"}
            else:
                matched = _REVISION.fullmatch(if_match)
                if matched is None:
                    raise PreconditionFailed(f"object revision changed: {key}")
                revision = int(matched.group(1)) + 1
                condition = {
                    "ConditionExpression": "#revision = :expected",
                    "ExpressionAttributeNames": {"#revision": "revision"},
                    "ExpressionAttributeValues": {":expected": {"N": matched.group(1)}},
                }
            self._write(lambda: self.dynamodb.put_item(
                TableName=self.aws.table,
                Item={"pk": {"S": full}, "data": {"B": payload}, "revision": {"N": str(revision)}},
                **condition,
            ))
            return StoredObject(payload, f"revision-{revision}")
        if if_match is not None:
            raise ValueError("S3 generation objects are create-only")
        checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
        response = self._write(lambda: self.s3.put_object(
            Bucket=self.aws.bucket, Key=full, Body=payload, IfNoneMatch="*",
            ChecksumSHA256=checksum,
        ))
        return StoredObject(payload, response["ETag"])

    def delete(self, key: str, *, if_match: str | None = None) -> bool:
        if self.read_only:
            raise ValueError("AWS reference store is read only")
        if _MUTABLE_KEY.fullmatch(key):
            raise ValueError("mutable records are never deleted so revisions cannot repeat")
        if if_match is not None:
            raise ValueError("conditional S3 deletes are not supported")
        # ponytail: existence check races a concurrent delete; nothing relies on the result.
        existed = self.object_size(key) is not None
        self.s3.delete_object(Bucket=self.aws.bucket, Key=self._full(key))
        return existed


def connect(value: str, *, read_only: bool = False) -> AwsObjectStore:
    """Build real clients from the ambient AWS credential chain (profile, SSO, role)."""
    location = parse_location(value)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "AWS reference storage needs boto3, e.g. `uv run --with boto3 trainguard ...`"
        ) from exc
    config = Config(
        region_name=location.region, connect_timeout=10, read_timeout=60,
        retries={"mode": "standard", "max_attempts": 3},
    )
    session = boto3.session.Session()
    return AwsObjectStore(
        session.client("s3", config=config), session.client("dynamodb", config=config),
        location, read_only=read_only,
    )
