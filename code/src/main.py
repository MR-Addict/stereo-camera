import cv2

from vision import predict, match_objects, draw_object


def main():
    # Open cameras
    left_cam = cv2.VideoCapture(1)
    right_cam = cv2.VideoCapture(2)

    if not left_cam.isOpened() or not right_cam.isOpened():
        print("Error: Cannot open cameras")
        return

    while True:

        ret1, left_frame = left_cam.read()
        ret2, right_frame = right_cam.read()

        if not ret1 or not ret2:
            print("Error: Cannot read frames from cameras")
            break

        # Predict on frames
        left_objects = predict(left_frame)
        right_objects = predict(right_frame)

        # Match the objects from the left and right frames
        matched_objects = match_objects(
            (left_frame, left_objects), (right_frame, right_objects)
        )

        # Draw the matched objects on the frames
        for obj in matched_objects:
            left_frame = draw_object(left_frame, obj[0])
            right_frame = draw_object(right_frame, obj[1])

        # Combine frames horizontally
        combined_frame = cv2.hconcat([left_frame, right_frame])

        # Display the combined frame
        cv2.imshow("Stereo Cameras", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the cameras and close windows
    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
