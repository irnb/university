# مباحثی در علوم کامپیوتر 🎓

این دوره با تشخیص استاد میتواند با تمرکز بر روی یک مبحث خاص از علوم کامپیوتر باشد و دوره ما با تشخیص استاد با تمرکز بر هوش مصنوعی و یادگیری ماشین خواهد بود.

## اطلاعات دوره 📚

- 👨‍🏫 مدرس: استاد میرخان
- 📖 کتاب رفرنس: Beginning Deep Learning with TensorFlow (Liangqu Long, Xianming Zeng)


## تمرین اول: پرسپترون 🤖

The perceptron is a type of machine learning algorithm for **supervised learning** of **binary classifiers**.


### مسئله 📝

فرض کنید یک مجموعه داده شامل چهار نقطه دو‌بُعدی داریم که باید با استفاده از پرسپترون در یادگیری ماشین نظارت‌شده طبقه‌بندی شوند.

**کلاس مثبت (+1):**
- $p_1 = (2, 3)$, $y_1 = +1$
- $p_2 = (3, 5)$, $y_2 = +1$

**کلاس منفی (-1):**
- $p_3 = (-1, -1)$, $y_3 = -1$
- $p_4 = (-2, -3)$, $y_4 = -1$

هدف ما پیدا کردن وزن‌ها ($w$) و بایاس ($b$) پرسپترون است که بتواند این نقاط را به درستی طبقه‌بندی کند.

---

### راه‌حل با محاسبات ریاضی ✏️

1. **مقداردهی اولیه:**

   w = [0, 0], b = 0

2. **الگوریتم پرسپترون:**

   الگوریتم پرسپترون به صورت تکراری وزن‌ها و بایاس را به‌روزرسانی می‌کند تا زمانی که تمام نقاط به درستی طبقه‌بندی شوند.

3. **قانون به‌روزرسانی:**

   `w_new = w_old + y_i * x_i`
   
   `b_new = b_old + y_i`

4. **تکرارها:**

   **تکرار اول:**

   - **نقطه p₁:**

     `output = sign(w·x₁ + b) = sign(0 + 0) = 0`

     چون 0 ≠ y₁، نیاز به به‌روزرسانی داریم.

     `w = [0, 0] + 1 * [2, 3] = [2, 3]`
     
     `b = 0 + 1 = +1`

   - **نقطه p₂:**

     `output = sign(w·x₂ + b) = sign(2×3 + 3×5 + 1) = sign(22) = +1`

   - **نقطه p₃:**

     `output = sign(w·x₃ + b) = sign(2×(-1) + 3×(-1) + 1) = sign(-4) = -1`

   - **نقطه p₄:**

     `output = sign(w·x₄ + b) = sign(2×(-2) + 3×(-3) + 1) = sign(-12) = -1`

تمام نقاط به درستی طبقه‌بندی شدند، بنابراین الگوریتم متوقف می‌شود.

---

### نتیجه 🎯

وزن‌های نهایی:
w = [2, 3]

بایاس نهایی:
b = +1

---

### کد پایتون 💻

```python
import numpy as np

# داده‌ها و برچسب‌ها
X = np.array([[2, 3],
              [3, 5],
              [-1, -1],
              [-2, -3]])
y = np.array([1, 1, -1, -1])

# مقداردهی اولیه وزن‌ها و بایاس
w = np.zeros(2)
b = 0
learning_rate = 1

# الگوریتم پرسپترون
for epoch in range(10):
    errors = 0
    for xi, target in zip(X, y):
        output = np.sign(np.dot(w, xi) + b)
        if output != target:
            w += learning_rate * target * xi
            b += learning_rate * target
            errors += 1
    if errors == 0:
        break

print("وزن‌های نهایی:", w)
print("بایاس نهایی:", b)

## Program Output 📤
## وزن‌های نهایی: [2. 3.]
## بایاس نهایی: 1
```


## تمرین دوم:  classifying handwritten digits 🧠

کد تمرین در فایل classifying_handwritten_digits.py قرار گرفته و با استفاده از دستورات زیر میتوانید اجرا کنید برنامه را


```
python3.12 -m venv .venv   
source .venv/bin/activate
pip3 install -r requirements.txt  

python3.12 classifying_handwritten_digits.py
```

این برنامه از MINST دیتا گرفته و سپس به کلاستر بندی تصاویر اعداد دست نوشته شده میپردازد دبا کمک نورال نتوورک

## کنفرانس: Practical AI 🤖

- 📝 NotebookLM
- 💬 LocalLLM (LMStudio, LAMA, API)
- 🔄 OpenAI Standard API 
- 🎨 Image Generation (Photo AI Idea)