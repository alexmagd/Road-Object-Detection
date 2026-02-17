from ultralytics import YOLO

def main():

    model = YOLO("yolov8l.yaml")

    results = model.train(data=r"Thesis\RoadObjectDetection\dataset\data.yaml", epochs=20)


if __name__ == '__main__':
    main()