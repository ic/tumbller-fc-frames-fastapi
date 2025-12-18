from base64 import b64encode


def make_jfs_header(fid: str, key: str) -> str:
    header = {
        "fid": fid,
        "type": "custody",
        "key": key,
    }
    return b64encode(bytes(str(header).encode('utf-8'))).decode('utf-8')

def make_jfs_payload(fqdn: str) -> str:
    payload = {
        "domain": fqdn,
    }
    return b64encode(bytes(str(payload).encode('utf-8'))).decode('utf-8')

def make_jfs_signature() -> str:
    return ""
