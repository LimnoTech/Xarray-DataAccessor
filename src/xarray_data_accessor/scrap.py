def _round_to_nearest(
    number: float,
    shift_up: bool,
) -> float:
    """Rounds number to nearest 0.25. Either shifts up or down."""
    num = round((number * 4)) / 4
    if shift_up:
        if num < number:
            num += 0.25
    else:
        if num > number:
            num -= 0.25
    return num


def _standardize_bbox(
    bbox: BoundingBoxDict,
) -> BoundingBoxDict:
    """
    Converts a bbox to the nearest 0.25 increments.

    NOTE: This is used when combining CDS and AWS data since CDS API returns
        data at 0.25 increments starting/stopping at the exact bbox coords.
        In contrast, AWS returns all data in 0.25 increments for the whole 
        globe and is converted via AWSDataAccessor._crop_aws_data().
    """
    out_bbox = {}

    out_bbox['west'] = _round_to_nearest(bbox['west'], shift_up=False)
    out_bbox['south'] = _round_to_nearest(bbox['south'], shift_up=False)
    out_bbox['east'] = _round_to_nearest(bbox['east'], shift_up=True)
    out_bbox['north'] = _round_to_nearest(bbox['north'], shift_up=True)

    return out_bbox
