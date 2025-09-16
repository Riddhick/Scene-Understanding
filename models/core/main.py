from detection import load_model, run_detection
from relationships import build_scene_graph
from distances import draw_pixel_distances
from visualization import draw_scene_graph, show_image
from object_metrics import compute_object_metrics, draw_object_metrics, normalize_distance
from scene_json import build_scene_json, save_scene_json

def main():
    img_path = "D:\Work\RCI\Code\Sample\9999986_00000_d_0000024.jpg"
    model = load_model()

    # Detection
    img, detected_objects = run_detection(model, img_path)

    # Scene graph
    scene_graph = build_scene_graph(detected_objects)

    # Draw distances
    img_with_distances = draw_pixel_distances(img, detected_objects)

    # Draw scene graph
    img_with_graph = draw_scene_graph(img_with_distances, scene_graph)

    # Show result
    #show_image(img_with_graph, "Scene Graph Visualization")
    metrics = compute_object_metrics(img, detected_objects)
    for m in metrics:
        print(m)  # optional: see values in console
    normalized_metric = normalize_distance(metrics,detected_objects)
    for m in normalized_metric:
        print(m)
# Draw metrics on image
    img_with_metrics = draw_object_metrics(img, metrics, detected_objects)

# Show annotated result
    show_image(img_with_metrics, "Scene Graph + Object Metrics")
    # Save JSON
    scene_json = build_scene_json(detected_objects, scene_graph)
    print(scene_json)
    #save_scene_json(scene_json)

if __name__ == "__main__":
    main()
