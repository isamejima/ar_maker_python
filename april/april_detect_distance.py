import copy
import time
import math
import cv2 as cv
from pupil_apriltags import Detector
import numpy as np

def main():
    families = 'tag36h11'
    nthreads = 1
    quad_decimate = 2.0
    quad_sigma = 0.0
    refine_edges = 1
    decode_sharpening = 0.25
    debug = 0

    # カメラ
    cap = cv.VideoCapture(2,cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_SETTINGS, 1)
    fov=78
    width=1280
    height=720
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, 60)

    # calc Camera Intrinsic Parameter Matrix
    fx = 1.0 / (2.0 * math.tan(np.radians(fov) / 2.0)) * width
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    mtx = np.asarray([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
    ])
    my_camera_params=(fx,fy,cx,cy)
    my_tag_size = 0.10 #in meter  

    # Detector
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        # カメラキャプチャ
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 検出実施
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            #estimate_tag_pose=False,
            estimate_tag_pose=True,         
            camera_params=(fx,fy,cx,cy),
            tag_size=my_tag_size,
        )

        # 描画
        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        # キー処理(ESC：終了)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映
        cv.imshow('AprilTag Detect Sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners
        pose_t = tag.pose_t
               
        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # 中心
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        # 各辺
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # タグファミリー、タグID        
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)        

        np.set_printoptions(precision=2)
        cv.putText(image,
           "id:" +str(tag_id)+ str(pose_t),
           (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
           cv.LINE_AA)
        

    # 処理時間
    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()