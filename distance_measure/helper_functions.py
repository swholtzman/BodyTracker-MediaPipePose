


def get_distance(self, width_px, height_px):
    """
    Calculates distance using both Width and Height.
    Returns the MINIMUM distance to filter out rotation spikes.

    Logic:
    - If we rotate (Yaw), 'width_px' shrinks -> 'dist_w' becomes HUGE (Error).
    - 'height_px' stays roughly same -> 'dist_h' stays Accurate.
    - min(Huge, Accurate) = Accurate.
    """
    dist_w = float('inf')
    dist_h = float('inf')

    if self.focal_ratio_width is not None and width_px > 0:
        dist_w = self.focal_ratio_width / width_px

    if self.focal_ratio_height is not None and height_px > 0:
        dist_h = self.focal_ratio_height / height_px

    # If both are invalid, return 0
    if dist_w == float('inf') and dist_h == float('inf'):
        return 0

    # Return the smaller distance (closest to camera), which filters out
    # the "infinite distance" artifact caused by rotation.
    return min(dist_w, dist_h)