import re

class LocalAgreement:
    def __init__(self, n=2):
        self.n = n
        self.history = []  
        
    def process(self, new_text: str) -> tuple[str, str]:
        """
        Trả về (stable_text, unstable_text)
        - stable_text: Phần văn bản đã chốt (dùng để Dịch và lưu)
        - unstable_text: Phần văn bản đang nhảy (dùng để hiển thị mờ)
        """
        if not new_text:
            return "", ""
            
        # 1. Chuẩn hóa & Tách từ (Word-level)
        # Xóa khoảng trắng thừa và tách theo dấu cách
        words = new_text.strip().split()
        
        self.history.append(words)
        
        # Giữ history đúng độ dài n
        if len(self.history) > self.n:
            self.history = self.history[-self.n:]
            
        if len(self.history) < self.n:
            return "", new_text

        # 2. Tìm phần chung dài nhất (Longest Common Prefix) theo TỪ
        min_len = min(len(sent) for sent in self.history)
        common_words = []
        
        for i in range(min_len):
            # Lấy từ thứ i của tất cả các phiên bản trong history
            current_words = [sent[i] for sent in self.history]
            
            # Kiểm tra xem tất cả có giống nhau không
            if all(w == current_words[0] for w in current_words):
                common_words.append(current_words[0])
            else:
                break
                
        # 3. Kết quả
        stable_text = " ".join(common_words)
        
        if stable_text:
            # Tìm vị trí kết thúc của stable_text trong new_text
            # Cộng thêm 1 để bỏ qua dấu cách
            unstable_start = len(stable_text)
            unstable_text = new_text[unstable_start:].strip()
        else:
            unstable_text = new_text

        return stable_text, unstable_text

    def reset(self):
        self.history = []
