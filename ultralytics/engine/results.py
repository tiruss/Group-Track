# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import json
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from pathlib import Path

class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        


        return self.__class__(self.data[idx], self.orig_shape)
    
    
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


count_colors = Colors()  # create instance for 'from utils.plots import colors'

final_outcomes=[]
class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Args:
        orig_img (numpy.ndarray): The original image as a numpy array.
        path (str): The path to the image file.
        names (dict): A dictionary of class names.
        boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
        masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
        probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.
 
    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (tuple): The original image shape in (height, width) format.
        boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
        masks (Masks, optional): A Masks object containing the detection masks.
        probs (Probs, optional): A Probs object containing probabilities of each class for classification task.
        keypoints (Keypoints, optional): A Keypoints object containing detected keypoints for each object.
        speed (dict): A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
        names (dict): A dictionary of class names.
        path (str): The path to the image file.
        _keys (tuple): A tuple of attribute names for non-empty attributes.
    """


    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None) -> None:
        """Initialize the Results class."""
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}  # milliseconds per image
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = 'boxes', 'masks', 'probs', 'keypoints'
        self.final_keys=None
        self.group_list=None
        self.img_count=None
        self.frame_count= None



    
    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply('__getitem__', idx)

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs

    def _apply(self, fn, *args, **kwargs):

        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        return self._apply('cpu')

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        return self._apply('numpy')

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        return self._apply('cuda')

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        return self._apply('to', *args, **kwargs)

    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def plot(
        
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font='Arial.ttf',
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        img_count=None,
        group_count = None,
        group_list=None,
        frame_count=None
        
        
    ):


        
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names)

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)


   
        self.final_keys=list(group_count.keys())
        self.viedo_count=frame_count
        self.img_count=img_count
        self.group_list=group_list
        if pred_boxes and show_boxes:
                unique_classes = pred_boxes.cls.unique()
                ###영상에서 사람이 디텍션이 안되고 그룹만 됬을경우 상황의 코드 추가
                if 0 not in unique_classes:
                    print("There are no boxes with person, only group exists.")
                else:
                    pred_ids = pred_boxes.id[pred_boxes.cls==0].to(int)
                    matching_person= [f[0] for f in self.final_keys]
                    for i, pid in reversed(list(enumerate(pred_ids))):
                        findx = self.final_keys[matching_person.index(pid)][1]
                       
                        if findx in group_list[0]:
                            index = group_list[0].index(findx)
                            mapped_value = group_list[1][index]
                            # The corresponding group ID
                            label = f'pid:{pid} Gid:{mapped_value}'

                            # 그룹일 때 지정된 색상 사용
                            color = count_colors(i, True)
                        else:
                            label = f'pid:{pid} Gid:solo'
                     
                            # solo일 때 검은색 사용
                            color = (0, 0, 0)  # 검은색으로 설정
                        # annotator.box_label 함수에 색상과 라벨 정보 전달
                        annotator.box_label(pred_boxes[i].xyxy.squeeze(), label, color=color)


                
                # # Plot Classify results 
                if pred_probs is not None and show_probs:
                    text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
                    x = round(self.orig_shape[0] * 0.03)
                    annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

                # Plot Pose results
                if self.keypoints is not None:
                    for k in reversed(self.keypoints.data):
                        annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()

    
    def verbose(self):
        """Return log string for each task."""
        log_string = ''
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f'{log_string}(no detections), '
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        final_keys=self.final_keys
        boxes = self.boxes
        outcomes = []
        group_ids = set()
        group_list=self.group_list
        if boxes:
            unique_classes = boxes.cls.unique()
            if 0 not in unique_classes:
                print("There are no boxes with person, only group exists.")
            else:
                
                pred_ids = boxes.id[boxes.cls == 0].to(int) # 사람에 해당하는 객체 ID
                matching_person = [f[0] for f in final_keys]  # 최종 키에서 사람 ID 추출
                for i, pid in reversed(list(enumerate(pred_ids))):
                    findx = final_keys[matching_person.index(pid)][1]
                    
                    if findx and findx in group_list[0]:
                        index = group_list[0].index(findx)
                        mapped_value = group_list[1][index]
                        outcomes.append('0')  # '0' for person
                        group_ids.add(mapped_value)
                    else:
                        outcomes.append('0')  # '0' for solo

        # Prepare the directory
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)

        # Save to txt
        with open(txt_file, 'w') as file:
            for outcome in outcomes:
                file.write(f'{outcome}\n')
            for gid in group_ids:
                file.write('1\n')  

        print(f"Results saved to {txt_file}")
                
                
    def save_json(self, jsonfile):
        
        if self.viedo_count ==0:
        
            frame = self.img_count 
        else :
            
            frame= self.viedo_count
        final_keys = self.final_keys
        group_list = self.group_list
        
        objects = []

        for boxes in self.boxes:
            if boxes.cls == 0:  # 'person' 객체에 대해서만 처리
                pid = boxes.id.item()
                bbox = boxes.xyxy.squeeze().tolist()
                
                # 매핑된 그룹 ID 찾기
                group_id = "solo"  # 기본값 설정
                for person_id, mapped_group_id in final_keys:
                    if pid == person_id:
                        if mapped_group_id in group_list[0]:
                            index = group_list[0].index(mapped_group_id)
                            group_id = group_list[1][index]  # 매핑된 그룹 ID 할당
                            break
                obj_info = {
                    "Type": "Person",
                    "ID": pid,
                    "Location": {
                        "Bounding Box": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3]
                        }
                    },
                    "Confidence Score": boxes.conf.item(),
                    "GroupID": group_id
                }
                objects.append(obj_info)

        # 객체 정보를 person_id 기준으로 정렬
        objects = sorted(objects, key=lambda x: x["ID"])

        # JSON 파일 경로
        json_file_path = Path(jsonfile) / 'inference_results.json'
        
        # 파일이 이미 존재하면 기존 데이터 불러오기
        if json_file_path.exists():
            with json_file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            # 기존 프레임 데이터에 현재 프레임 데이터 추가
            data.append({
                "frame": frame,
                "Objects": objects
            })
        else:
            # 파일이 존재하지 않으면 새로운 데이터 구조 생성
            data = [{
                "resolution": '720p',
                "width": 1280,
                "height": 720,
                "frame": frame,
                "Objects": objects
            }]
        
        # 변경된 데이터를 파일에 저장
        with json_file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


            
    def save_crop(self, save_dir, file_name=Path('im.jpg')):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        if self.probs is not None:
            LOGGER.warning('WARNING ⚠️ Classify task do not support `save_crop`.')
            return
        for d in self.boxes:
            save_one_box(d.xyxy,
                         self.orig_img.copy(),
                         file=Path(save_dir) / self.names[int(d.cls)] / f'{Path(file_name)}.jpg',
                         BGR=True)

    def tojson(self, normalize=False):
        """Convert the object to JSON format."""
        if self.probs is not None:
            LOGGER.warning('Warning: Classify task do not support `tojson` yet.')
            return

        import json

        # Create list of detection dictionaries
        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
            conf = row[-2]
            class_id = int(row[-1])
            name = self.names[class_id]
            result = {'name': name, 'class': class_id, 'confidence': conf, 'box': box}
            if self.boxes.is_track:
                result['track_id'] = int(row[-3])  # track ID
            if self.masks:
                x, y = self.masks.xy[i][:, 0], self.masks.xy[i][:, 1]  # numpy array
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
            results.append(result)

        # Convert detections to JSON
        return json.dumps(results, indent=2)


class Boxes(BaseTensor):
    """
    A class for storing and manipulating detection boxes.

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
            If present, the third last column contains track IDs.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        xyxy (torch.Tensor | numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xywh (torch.Tensor | numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor | numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor | numpy.ndarray): The boxes in xywh format normalized by original image size.
        data (torch.Tensor): The raw bboxes tensor (alias for `boxes`).

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """Initialize the Boxes class."""
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (6, 7), f'expected `n` in [6, 7], but got {n}'  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Attributes:
        xy (list): A list of segments in pixel coordinates.
        xyn (list): A list of normalized segments.

    Methods:
        cpu(): Returns the masks tensor on CPU memory.
        numpy(): Returns the masks tensor as a numpy array.
        cuda(): Returns the masks tensor on GPU memory.
        to(device, dtype): Returns the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initialize the Masks class with the given masks tensor and original image shape."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Return normalized segments."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Return segments in pixel coordinates."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    Attributes:
        xy (torch.Tensor): A collection of keypoints containing x, y coordinates for each detection.
        xyn (torch.Tensor): A normalized version of xy with coordinates in the range [0, 1].
        conf (torch.Tensor): Confidence values associated with keypoints if available, otherwise None.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(device, dtype): Returns a copy of the keypoints tensor with the specified device and dtype.
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error
    def __init__(self, keypoints, orig_shape) -> None:
        """Initializes the Keypoints object with detection keypoints and original image size."""
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Returns x, y coordinates of keypoints."""
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Returns normalized x, y coordinates of keypoints."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """Returns confidence values of keypoints if available, else None."""
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.

    Attributes:
        top1 (int): Index of the top 1 class.
        top5 (list[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor): Confidence of the top 1 class.
        top5conf (torch.Tensor): Confidences of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """Initialize the Probs class with classification probabilities and optional original shape of the image."""
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the index of top 1."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of top 5."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Return the confidence of top 1."""
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Return the confidences of top 5."""
        return self.data[self.top5]

                