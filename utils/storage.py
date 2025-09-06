from __future__ import annotations
from pathlib import Path
import os

class LocalDiskStorage:
    def save(self, src: str | Path, dest_rel: str) -> str:
        dest = Path("artifacts") / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(src).replace(dest)
        return str(dest)

def get_storage():
    # S3 support (optional) if env creds exist and boto3 installed
    if all(os.getenv(k) for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_REGION","AWS_S3_BUCKET")):
        try:
            import boto3  # not in requirements; optional
            class S3Storage:
                def __init__(self):
                    self.bucket = os.environ["AWS_S3_BUCKET"]
                    self.s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
                def save(self, src: str | Path, dest_rel: str) -> str:
                    self.s3.upload_file(str(src), self.bucket, dest_rel, ExtraArgs={"ACL":"public-read","ContentType":"video/mp4"})
                    return f"https://{self.bucket}.s3.amazonaws.com/{dest_rel}"
            return S3Storage()
        except Exception:
            pass
    return LocalDiskStorage()
