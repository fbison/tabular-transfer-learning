def parse_key(key):
    """
    key format: {type}_{id}_{imputation}
    """
    parts = key.split("_")

    return {
        "type": parts[0],          # upstream / downstream
        "id": parts[1],            # 2,3,4
        "imputation": parts[2]     # gaussian, mean, pseudo, gt
    }


def is_valid_pair(meta_a, meta_b):
    return (
        meta_a["type"] != meta_b["type"] and
        meta_a["id"] == meta_b["id"] and
        meta_a["imputation"] == meta_b["imputation"]
    )