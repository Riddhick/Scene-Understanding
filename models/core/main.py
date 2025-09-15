from detection import load_model, run_detection
from relationships import build_scene_graph
from distances import draw_pixel_distances
from visualization import draw_scene_graph, show_image
from scene_json import build_scene_json, save_scene_json

def main():
    img_path = "D:\Work\RCI\Code\Sample\\0000103_03738_d_0000032.jpg"
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
    show_image(img_with_graph, "Scene Graph Visualization")

    # Save JSON
    scene_json = build_scene_json(detected_objects, scene_graph)
    print(scene_json)
    #save_scene_json(scene_json)

if __name__ == "__main__":
    main()
