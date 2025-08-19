#!/usr/bin/env python3


import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import pipeline
import torch
import numpy as np
import cv2
from PIL import Image as PILImage


bridge = CvBridge()


class YoloDepthCalibrator:
    def __init__(self):
        rospy.init_node("yolo_depth_calibration_node", anonymous=True)


       
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )


        
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24
        )


        self.latest_rgb = None
        self.latest_depth = None
        self.depth_pairs = []


        self.device = self._setup_device()
        self.load_models()
        self._warmup_models()


    def _setup_device(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            device = torch.device('cuda')
            try:
                _ = torch.randn(1).to(device)
                rospy.loginfo("CUDA device available and initialized.")
                return device
            except Exception as e:
                rospy.logwarn(f"CUDA test failed: {e}, falling back to CPU.")
        return torch.device('cpu')


    def load_models(self):
        rospy.loginfo("Loading YOLO model...")
        self.yolo_model = YOLO('/home/oshah7/catkin_ws/src/yolo_depth_node/models/my_model.pt').to(self.device)
        rospy.loginfo(f"YOLO model loaded on device: {self.device}")


        rospy.loginfo("Loading Depth Anything model...")
        depth_device = 0 if self.device.type == 'cuda' else -1
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=depth_device
        )
        rospy.loginfo(f"Depth Anything model loaded on device: {depth_device}")


    def _warmup_models(self):
        rospy.loginfo("Warming up models...")
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 640, 640).to(self.device) / 255.0
            _ = self.yolo_model(dummy_img)
        dummy_pil = PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        _ = self.depth_estimator(dummy_pil)
        rospy.loginfo("Models warmed up successfully.")


    
    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_rgb = frame
        except Exception as e:
            rospy.logerr("Error decoding compressed image: %s", e)
            return
        self.process_frame()


    def depth_callback(self, msg):
        try:
            self.latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Depth conversion error: {e}")


    def process_frame(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return


        rgb = self.latest_rgb.copy()
        depth_img = self.latest_depth.copy()


        results = self.yolo_model(rgb)
        detections = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else np.array([])


        da_depth_map = self.run_depth_anything(rgb)


        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])
            conf = results[0].boxes.conf[i].item()
            cls = int(results[0].boxes.cls[i].item())


            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2


            if cx < 0 or cy < 0 or cx >= depth_img.shape[1] or cy >= depth_img.shape[0]:
                continue


            sensor_depth = depth_img[cy, cx]
            if isinstance(sensor_depth, np.uint16) or (hasattr(sensor_depth, 'dtype') and sensor_depth.dtype == np.uint16):
                sensor_depth = sensor_depth / 1000.0
            else:
                sensor_depth = float(sensor_depth)


            # Extract normalized DA depth from bounding box
            box_depth = da_depth_map[y1:y2, x1:x2]
            da_depth = np.median(box_depth)


            if 0.1 < sensor_depth < 10.0 and da_depth > 0:
                # Scale factor for per-frame calibration
                scale = sensor_depth / da_depth
                calibrated_depth_map = da_depth_map * scale  # Now in meters


                # Estimated depth at center
                est_depth = calibrated_depth_map[cy, cx]


                
                self.depth_pairs.append((da_depth, sensor_depth))


                
                label = f"GT: {sensor_depth:.2f}m | Cal: {est_depth:.2f}m"
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        cv2.imshow("YOLO + Calibrated Depth", rgb)
        cv2.waitKey(1)


    def run_depth_anything(self, bgr_np):
        rgb_pil = PILImage.fromarray(cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB))
        result = self.depth_estimator(rgb_pil)
        depth_map = np.array(result["depth"])
        depth_map = cv2.resize(depth_map, (bgr_np.shape[1], bgr_np.shape[0]))


        # Normalized to [0, 1]
        normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())


        #Inverted so closer = smaller
        inverted = 1.0 - normalized


        return inverted




    def save_depth_pairs(self):
        np.save("depth_pairs.npy", np.array(self.depth_pairs))
        rospy.loginfo("Saved depth calibration data to depth_pairs.npy.")


if __name__ == '__main__':
    try:
        calibrator = YoloDepthCalibrator()
        rospy.on_shutdown(calibrator.save_depth_pairs)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
