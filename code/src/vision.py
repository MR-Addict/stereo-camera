import cv2
import torch
from ultralytics import YOLO

# check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load offical model
model = YOLO("models/yolov8x.pt")

# move the model to the chosen device
model.to(device)


def predict(frame, conf=0.6, verbose=False):
    # predict
    result = model.predict(frame, conf=conf, verbose=verbose, classes=[39])[0]

    # prediction result
    objects = []

    for box in result.boxes:
        # label
        label = result.names[int(box.cls.tolist()[0])]
        # confidence
        conf = f"{box.conf.tolist()[0]:.2f}"
        # x1, y1, x2, y2
        x1, y1, x2, y2 = [int(i) for i in box.xyxy.tolist()[0]]
        # append to data
        objects.append((label, conf, (x1, y1, x2, y2)))

    return objects


def draw_object(frame, obj):
    # draw the objects on the frame
    label, conf, (x1, y1, x2, y2) = obj
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{label} {conf}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return frame


def compare_similarity(left, right):
    (left_frame, left_object), (right_frame, right_object) = left, right

    """
    Method 1: Compare the similarity by histograms
    """

    # get the object from the left frame
    left_x1, left_y1, left_x2, left_y2 = left_object[2]
    left_obj = left_frame[left_y1:left_y2, left_x1:left_x2]

    # get the object from the right frame
    right_x1, right_y1, right_x2, right_y2 = right_object[2]
    right_obj = right_frame[right_y1:right_y2, right_x1:right_x2]

    first_hist = cv2.calcHist([left_obj], [0], None, [256], [0, 256])
    second_hist = cv2.calcHist([right_obj], [0], None, [256], [0, 256])

    hist_similarity = 1 - cv2.compareHist(
        first_hist, second_hist, cv2.HISTCMP_BHATTACHARYYA
    )

    """
    Method 2: compare the similarity by area overlap
    """

    # get the area of the object
    left_area = left_obj.shape[0] * left_obj.shape[1]
    right_area = right_obj.shape[0] * right_obj.shape[1]

    # get area similarity
    area_similarity = 1 - (abs(left_area - right_area) / left_area)

    """
    Method 3: compare the similarity by x and y coordinates
    """

    # get the x and y similarity
    x_similarity = 1 - abs(left_x1 - right_x2) / left_frame.shape[1]
    y_similarity = 1 - abs(left_y1 - right_y1) / left_frame.shape[0]

    similarity = (
        hist_similarity * 0.2
        + x_similarity * 0.25
        + y_similarity * 0.25
        + area_similarity * 0.3
    )

    print(similarity)

    # return the similarity
    return similarity


def match_objects(left, right, conf=0.6):
    # match the objects from the left and right frames
    left_frame, left_objects = left
    right_frame, right_objects = right

    # matched objects
    matched_objects = []

    # compare objects
    for left_object in left_objects:
        objs_with_similarity = []
        objs_with_same_label = [a for a in right_objects if a[0] == left_object[0]]

        # skip if no objects with the same label
        if not objs_with_same_label:
            continue

        for right_object in objs_with_same_label:
            # compare the similarity
            similarity = compare_similarity(
                (left_frame, left_object), (right_frame, right_object)
            )

            # append to the list
            objs_with_similarity.append((right_object, similarity))

        # sort the list by similarity
        objs_with_similarity.sort(key=lambda x: x[1])

        # get the most similar object
        first_obj = objs_with_similarity[0]
        if first_obj[1] >= conf:
            matched_objects.append((left_object, first_obj[0]))

    return matched_objects
