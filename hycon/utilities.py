from floris.utilities import wrap_180


def convert_absolute_nacelle_heading_to_offset(target_nac_heading, current_nac_heading):
    # NOTE: by convention, absolute headings are given CW positive, but offsets
    # are given CCW positive.

    return -1 * wrap_180(target_nac_heading - current_nac_heading)
