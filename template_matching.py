import numpy as np
import cv2


class TemplateMatching:
    def __init__(self, initial, rect):
        self.initial = initial
        self.rect = rect

    def track(self, next_frame):
        if self.rect is None:
            return None
        # print(self.rect)
        query = self.initial[self.rect[1] : self.rect[3], self.rect[0] : self.rect[2]]
        # print(query)
        self.initial = next_frame
        temp_height, temp_width, _ = query.shape
        next_rect = (
            int(self.rect[0] - temp_width / 2),
            int(self.rect[1] - temp_height / 2),
            int(self.rect[2] + temp_width / 2),
            int(self.rect[3] + temp_height / 2),
        )
        next_left_margin = min(self.rect[0], temp_width / 2)
        next_right_margin = min(self.rect[1], temp_height / 2)
        next_rect = self.check_bounds(next_frame.shape, next_rect)
        next_subframe = next_frame[
            next_rect[1] : next_rect[3], next_rect[0] : next_rect[2]
        ]
        mt = cv2.matchTemplate(next_subframe, query, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(mt)
        if max_val < 0.6:
            self.rect = None
            return None
        result_rect = (
            int(self.rect[0] - next_left_margin + max_loc[0]),
            int(self.rect[1] - next_right_margin + max_loc[1]),
            int(self.rect[2] - next_left_margin + max_loc[0]),
            int(self.rect[3] - next_right_margin + max_loc[1]),
        )
        self.rect = self.check_bounds(next_frame.shape, result_rect)
        return self.rect

    @classmethod
    def check_bounds(cls, img_shape, rect):
        height, width, _ = img_shape
        left, top, right, bottom = rect
        left = max(left, 0)
        top = max(top, 0)
        right = min(width, right)
        bottom = min(height, bottom)
        return left, top, right, bottom
