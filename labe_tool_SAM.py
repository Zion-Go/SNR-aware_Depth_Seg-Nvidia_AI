import os
import cv2 as cv
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def get_mask_predictor(sam_checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
def mouse_callback(event, x, y, flags, params):
    pre_pt = params[3]
    if pre_pt == [x, y]: return
    image = params[0]
    # 在image上修改的可以image=xxx 否则就要使用params[0]=xxx
    predictor = params[1]
    win_name = params[2]
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(image, (x, y), 3, (255, 255, 255), -1, cv.LINE_AA)
        input_point = np.array([[x, y]], dtype=np.int32)
        input_label = np.array([1])
        masks, scores = one_point(input_point, input_label, predictor)
        p1, p2 = generatr_box_from_mask(image, masks)
        # 使用p1, p2与已经存在的box进行对比, 如果面积相近的话就视为同一个而不进行增加
        # params[0] = apply_mask_on_image(image, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
        cv.imshow(win_name, params[0])
    elif event == cv.EVENT_MOUSEMOVE:
        # 这个肯定不是画在原图上
        input_point = np.array([[x, y]], dtype=np.int32)
        input_label = np.array([1])
        masks, scores = one_point(input_point, input_label, predictor)
        image_copy = image.copy()
        p1, p2 = generatr_box_from_mask(image_copy, masks)
        image_copy = apply_mask_on_image(image_copy, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
        cv.imshow(win_name, image_copy)

ratio = 2
file = None
def write_data(p1, p2, img):
    h, w, c = img.shape
    width, height = p2[0] - p1[0], p2[1] - p1[1]
    center_x, center_y = p1[0] + width / 2, p2[1] + height / 2
    # 在这里面写入你想写的文件
    # 上面的这四个值需要除原图像的center_x /宽， width / 宽 ，center_y / 高 ，height / 高 这样就是0-1之间的数据
    print(f"0 {center_x / w * ratio} {center_y / h * ratio} {width / w * ratio} {height / h * ratio}")
    file.write(f"0 {center_x / w * ratio} {center_y / h * ratio} {width / w * ratio} {height / h * ratio}\n")

def mouse_callback2(event, x, y, flags, params):
    image = params[0]
    pts = params[3]
    # 在image上修改的可以image=xxx 否则就要使用params[0]=xxx
    predictor = params[1]
    win_name = params[2]
    if len(pts) == 0:
        if event == cv.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            cv.circle(image, (x, y), 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.imshow(win_name, params[0])
    elif len(pts) == 1:
        if event == cv.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            input_point = np.array(pts, dtype=np.int32)
            input_label = np.array([1, 1])
            masks, scores = mul_point(input_point, input_label, predictor)
            p1, p2 = generatr_box_from_mask(image, masks)
            write_data(p1, p2, params[0])
            # 数据在这

            # 使用p1, p2与已经存在的box进行对比, 如果面积相近的话就视为同一个而不进行增加
            # params[0] = apply_mask_on_image(image, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
            cv.imshow(win_name, params[0])
            pts.clear()
        elif event == cv.EVENT_MOUSEMOVE:
            input_point = np.array(pts + [[x, y]], dtype=np.int32)
            input_label = np.array([1, 1])
            masks, scores = mul_point(input_point, input_label, predictor)
            image_copy = image.copy()
            p1, p2 = generatr_box_from_mask(image_copy, masks)
            image_copy = apply_mask_on_image(image_copy, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
            cv.imshow(win_name, image_copy)

def mul_point(input_point, input_label, predictor):
    # 多个点 来确定某个物体
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    masks, scores = process_mask(masks, scores)
    return masks, scores
def generatr_box_from_mask(image, mask, random_color=True):
    if random_color:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    else:
        color = (30, 144, 255)
    # 其实也可以不利用opencv实现 支付要确定mask的四个边界就可以了
    matrix = mask.squeeze().astype(np.uint8) * 255
    # 寻找轮廓
    contours, _ = cv.findContours(matrix, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 最小外接矩形 带有旋转矫正的
    # rect = cv.minAreaRect(contours[0])
    # box = np.array(cv.boxPoints(rect), dtype=np.int32)
    # cv.drawContours(image, [box], 0, (0, 255, 0), 2, cv.LINE_AA)
    # 正的外接矩形 不带有自动矫正
    x_min = matrix.shape[1] - 1
    y_min = matrix.shape[0] - 1
    x_max = 0
    y_max = 0
    for cnt in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv.boundingRect(cnt)
        # 更新最大外接矩形的坐标
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2, cv.LINE_AA)
    return (x_min, y_min), (x_max, y_max)
def get_image(image_path):
    image = cv.imread(image_path)
    image = cv.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)))
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
def process_mask(masks, scores):
    idx = np.argmax(scores, axis=-1)
    if len(masks.shape) == 4:
        # 多个批量
        return masks[np.arange(masks.shape[0]), idx], scores[np.arange(scores.shape[0]), idx]
    else:
        return masks[idx][None], scores[idx]
def apply_mask_on_image(image, mask, random_color=True):
    if random_color:
        color = np.concatenate([np.random.randint(0, 255, 3, dtype=np.uint8)], axis=0)
    else:
        color = np.array([30, 144, 255], dtype=np.uint8)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    return cv.addWeighted(image, 1, mask_image, 0.65, 0)
def one_point(input_point, input_label, predictor):
    # 一个点
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    masks, scores = process_mask(masks, scores)
    return masks, scores
def one_point_example(predictor, image_path, win_name="1"):
    pre_pt = [0, 0]
    image = get_image(image_path)
    predictor.set_image(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    window = cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, mouse_callback, param=[image, predictor, win_name, pre_pt])
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def two_point_example(predictor, image_path, win_name="1"):
    image = get_image(image_path)
    predictor.set_image(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    window = cv.namedWindow(win_name)
    pts = []
    cv.setMouseCallback(win_name, mouse_callback2, param=[image, predictor, win_name, pts])
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def get_predictor(checkpoint, model_type, device):
    predictor = get_mask_predictor(checkpoint, model_type, device)
    return predictor
def main():
    global file
    predictor = get_predictor("checkpoint/sam_vit_l_0b3195.pth", "vit_l", "cuda")
    for image in os.listdir('images'):
        file = open(os.path.join('labels', image.split('.')[0] + '.txt'), mode='w', encoding='utf-8')
        two_point_example(predictor, os.path.join('images', image))
        file.close()

if __name__ == '__main__':
    main()
