from detection import load_model, run_detection
from spatialrel import build_scene_graph
from distances import draw_pixel_distances
from visualization import draw_scene_graph, show_image, draw_object_angles, save_image
from object_metrics import compute_object_metrics, draw_object_metrics, normalize_distance
from scene_json import build_scene_json, save_scene_json
from semantic_feature import add_semantic_features
from semantic_feature_parallel import add_semantic_features_hybrid
import cv2


def main():
    img_path = "D:\\Work\\RCI\\Code\\Sample\\0000103_03738_d_0000032.jpg"
    model = load_model()

    # Detection
    img, detected_objects = run_detection(model, img_path)
    #detected_objects = add_semantic_features(img, detected_objects,debug=True)
    detected_objects  = add_semantic_features_hybrid(img, detected_objects,debug = True)
    # Scene graph
    scene_graph = build_scene_graph(detected_objects)

    # Draw distances
    img_with_distances = draw_pixel_distances(img, detected_objects)

    # Draw scene graph
    img_with_graph = draw_scene_graph(img, scene_graph)
    img_with_angles = draw_object_angles(img, detected_objects, scene_graph)
    # Show the new visualization
    #show_image(img_with_angles, "Object Angles Visualization")
    # Show result
    #show_image(img, "Graph Visualization")
    #save_image(img,"graph4.png")
    metrics = compute_object_metrics(img, detected_objects)
    #for m in metrics:
        #print(m)  # optional: see values in console
    #normalized_metric = normalize_distance(metrics,detected_objects)
    #for m in normalized_metric:
        #print(m)
# Draw metrics on image
    img_with_metrics = draw_object_metrics(img, metrics, detected_objects)

# Show annotated result
    show_image(img_with_metrics, "Scene Graph + Object Metrics")
    # Save JSON
    scene_json = build_scene_json(detected_objects, scene_graph)
    print(scene_json)
    save_scene_json(scene_json)

if __name__ == "__main__":
    main()
