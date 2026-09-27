# AWS S3/DynamoDB 参考存储实测准备

日期：2026-09-27。适用分支：`feature/real-infra-adapters`。本文件不改变执行输入，也不授权生产使用。

## 1. 这一步要证明什么

`src/trainguard/aws_store.py` 把检查点协议接到 S3 与 DynamoDB：

- 生成代次载荷写入 S3，只用 `If-None-Match: *` 创建，并附 SHA-256 上传校验和；
- HEAD 与 AUTHORITY 存在 DynamoDB，用每键单调递增的 `revision` 做条件写，强一致读取，永不删除。S3 ETag 是内容摘要，A→B→A 后会复用，因此不能充当 HEAD 的 CAS 令牌。

本机只用进程内替身验证过映射逻辑。真实服务上需要重新确认：条件写 412/409、条件检查失败、强一致读、分页列举、并发 CAS 只有一个胜者、写后丢响应的回读、坏对象回退、接管后旧身份被拒，以及两 rank CPU 训练在删除本地检查点后从 HEAD 精确恢复。

## 2. 由你完成的账号准备

凭据由你配置（`aws configure` 或 `aws configure sso`），脚本只使用本机凭据链，不读取也不需要密钥内容。建议先在账单控制台设置预算告警（例如每月 20 美元）。

```bash
export AWS_REGION=us-east-1
export TG_BUCKET=trainguard-validation-<你的账号或随机后缀>
export TG_TABLE=trainguard-heads

aws s3api create-bucket --bucket "$TG_BUCKET" --region "$AWS_REGION"
aws s3api put-public-access-block --bucket "$TG_BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws dynamodb create-table --table-name "$TG_TABLE" --region "$AWS_REGION" \
  --attribute-definitions AttributeName=pk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH --billing-mode PAY_PER_REQUEST
```

`us-east-1` 以外的区域创建桶时需加 `--create-bucket-configuration LocationConstraint=$AWS_REGION`。

实测身份的最小权限（`DeleteObject` 与 `DeleteItem` 只用于清理冒烟运行自己的前缀，训练运行不需要）：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::<bucket>/validation/*"},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::<bucket>",
     "Condition": {"StringLike": {"s3:prefix": "validation/*"}}},
    {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"],
     "Resource": "arn:aws:dynamodb:<region>:<account>:table/<table>"}
  ]
}
```

## 3. 实测命令

位置字符串必须是规范形式，参数顺序为 `region` 在前、`table` 在后：

```bash
uv run --with boto3 python scripts/aws_reference_smoke.py \
  --location "aws://$TG_BUCKET/validation?region=$AWS_REGION&table=$TG_TABLE" \
  --receipt runs/aws-smoke-$(date -u +%Y%m%dT%H%M%SZ).json --training
```

脚本每次使用新的运行 ID，结束后只删除该 ID 下的对象与记录。请求量为数百次级，费用预计在几美分量级（以账单为准）。收据记录每个场景的 `PASS`/`FAIL`、错误摘要、客户端版本和时间。

## 4. 通过后能说什么、不能说什么

- 能说：在所记录的桶、表、区域和凭据下，协议的条件写、回读、回退、接管隔离，以及两 rank CPU 训练经 S3/DynamoDB 的 HEAD 恢复按预期工作。
- 不能说：跨主机隔离、断电耐久、GPU/NCCL、多区域、其他对象服务、限流与凭据过期下的长期行为，或任何生产授权。这些仍分别由 `cross_host_fencing`、`real_gpu_matrix`、`persistent_checkpoint` 等门槛覆盖。`persistent_checkpoint` 还需把实测收据纳入受审证据与完整故障矩阵后才可能改为通过。
