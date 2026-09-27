"""In-process S3 and DynamoDB doubles with the response shapes the adapter reads.

They model conditional-write outcomes and injected faults only; they are not
evidence about the real services.
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading


class FakeClientError(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeTransportError(Exception):
    """A connection dropped before or after the service applied a request."""


class _Faults:
    def __init__(self) -> None:
        self.faults: dict[tuple[str, str], list[str]] = {}
        self.lock = threading.RLock()

    def inject(self, operation: str, key: str, *kinds: str) -> None:
        self.faults.setdefault((operation, key), []).extend(kinds)

    def _next(self, operation: str, key: str) -> str | None:
        queue = self.faults.get((operation, key))
        return queue.pop(0) if queue else None

    def _apply(self, operation: str, key: str, action):
        kind = self._next(operation, key)
        if kind == "before":
            raise FakeTransportError("connection dropped before the request applied")
        if kind == "throttle":
            raise FakeClientError("SlowDown", 503)
        if kind == "conflict":
            raise FakeClientError("ConditionalRequestConflict", 409)
        result = action()
        if kind == "after":
            raise FakeTransportError("connection dropped after the request applied")
        return result


class FakeS3(_Faults):
    def __init__(self, page_size: int = 2) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size

    @staticmethod
    def etag(data: bytes) -> str:
        return '"' + hashlib.md5(data, usedforsecurity=False).hexdigest() + '"'

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, ChecksumSHA256=None):
        del Bucket
        if IfNoneMatch not in {None, "*"}:
            raise FakeClientError("InvalidArgument", 400)
        if ChecksumSHA256 != base64.b64encode(hashlib.sha256(Body).digest()).decode():
            raise FakeClientError("BadDigest", 400)

        def action():
            with self.lock:
                if IfNoneMatch == "*" and Key in self.objects:
                    raise FakeClientError("PreconditionFailed", 412)
                self.objects[Key] = bytes(Body)
                return {"ETag": self.etag(Body)}

        return self._apply("put", Key, action)

    def get_object(self, *, Bucket, Key):
        del Bucket
        with self.lock:
            if Key not in self.objects:
                raise FakeClientError("NoSuchKey", 404)
            data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ETag": self.etag(data)}

    def head_object(self, *, Bucket, Key):
        del Bucket
        with self.lock:
            if Key not in self.objects:
                raise FakeClientError("404", 404)
            return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        del Bucket
        with self.lock:
            keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        truncated = start + self.page_size < len(keys)
        response = {"Contents": [{"Key": key} for key in page], "IsTruncated": truncated}
        if truncated:
            response["NextContinuationToken"] = str(start + self.page_size)
        return response

    def delete_object(self, *, Bucket, Key):
        del Bucket
        with self.lock:
            self.objects.pop(Key, None)
        return {}


class FakeDynamoDB(_Faults):
    def __init__(self) -> None:
        super().__init__()
        self.items: dict[str, dict] = {}

    def get_item(self, *, TableName, Key, ConsistentRead=False):
        del TableName
        if ConsistentRead is not True:
            raise AssertionError("the adapter must read HEAD and AUTHORITY strongly")
        with self.lock:
            item = self.items.get(Key["pk"]["S"])
        return {} if item is None else {"Item": dict(item)}

    def put_item(self, *, TableName, Item, ConditionExpression,
                 ExpressionAttributeNames=None, ExpressionAttributeValues=None):
        del TableName
        key = Item["pk"]["S"]

        def action():
            with self.lock:
                current = self.items.get(key)
                if ConditionExpression == "attribute_not_exists(pk)":
                    passed = current is None
                elif (
                    ConditionExpression == "#revision = :expected"
                    and ExpressionAttributeNames == {"#revision": "revision"}
                ):
                    passed = (
                        current is not None
                        and current["revision"] == ExpressionAttributeValues[":expected"]
                    )
                else:
                    raise FakeClientError("ValidationException", 400)
                if not passed:
                    raise FakeClientError("ConditionalCheckFailedException", 400)
                self.items[key] = dict(Item)
                return {}

        return self._apply("put", key, action)
