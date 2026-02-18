def compute_gsd_cm_per_pixel(
    altitude_m,
    sensor_width_mm,
    sensor_height_mm,
    focal_length_mm,
    image_width_px,
    image_height_px
):
    altitude_cm = altitude_m * 100.0
    sensor_width_cm = sensor_width_mm / 10.0
    sensor_height_cm = sensor_height_mm / 10.0
    focal_length_cm = focal_length_mm / 10.0

    gsd_x = (altitude_cm * sensor_width_cm) / (focal_length_cm * image_width_px)
    gsd_y = (altitude_cm * sensor_height_cm) / (focal_length_cm * image_height_px)

    return (gsd_x + gsd_y) / 2.0


def real_distance_to_pixels(value, unit, gsd_cm_per_pixel):
    if unit == "px":
        return float(value)

    if unit == "cm":
        value_cm = value
    elif unit == "m":
        value_cm = value * 100.0
    elif unit == "km":
        value_cm = value * 100000.0
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return value_cm / gsd_cm_per_pixel
