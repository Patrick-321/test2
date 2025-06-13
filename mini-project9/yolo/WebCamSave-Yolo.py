import torch
import cv2
import time
import numpy as np
import random
import math

# Load YOLOv5 model (small version for speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Target classes
TARGET_CLASSES = ['person', 'cell phone', 'cup']
SAVE_DURATION = 5  # seconds

# Fun activity variables
particles = []
magic_mode = False
effect_start_time = 0
EFFECT_DURATION = 3  # seconds


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-5, -1)
        self.life = 1.0
        self.color = color
        self.size = random.randint(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 0.02
        return self.life > 0

    def draw(self, frame):
        if self.life > 0:
            alpha = int(255 * self.life)
            cv2.circle(frame, (int(self.x), int(self.y)), self.size,
                       (*self.color, alpha), -1)


def add_sparkle_effect(frame, x, y, w, h):
    """Add sparkles around detected objects"""
    global particles

    # Add new particles around the object
    for _ in range(5):
        px = random.randint(x, x + w)
        py = random.randint(y, y + h)
        color = random.choice([(255, 215, 0), (255, 20, 147), (0, 255, 255), (255, 105, 180)])
        particles.append(Particle(px, py, color))

    # Update and draw existing particles
    particles = [p for p in particles if p.update()]
    for particle in particles:
        particle.draw(frame)


def add_rainbow_border(frame, x, y, w, h, time_offset=0):
    """Add animated rainbow border around detected object"""
    colors = [
        (255, 0, 0),  # Red
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (75, 0, 130),  # Indigo
        (148, 0, 211)  # Violet
    ]

    # Animate color selection based on time
    color_index = int((time.time() * 3 + time_offset) % len(colors))
    color = colors[color_index]

    # Draw thick animated border
    thickness = int(5 + 3 * math.sin(time.time() * 4 + time_offset))
    cv2.rectangle(frame, (x - thickness, y - thickness),
                  (x + w + thickness, y + h + thickness), color, thickness)


def add_floating_emojis(frame, x, y, w, h, object_type):
    """Add floating emojis based on detected object type"""
    emoji_map = {
        'person': '👋 🌟 ✨',
        'cell phone': '📱 💫 🔮',
        'book': '📚 🎓 ⭐'
    }

    emoji_text = emoji_map.get(object_type, '✨ 🌟 💫')

    # Floating animation
    float_offset = int(10 * math.sin(time.time() * 2))

    cv2.putText(frame, emoji_text,
                (x, y - 20 + float_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def add_magic_text_effects(frame):
    """Add magical text effects when objects are detected"""
    messages = [
        "✨ MAGIC DETECTED! ✨",
        "🌟 ENCHANTED VISION 🌟",
        "🔮 MYSTICAL POWERS ACTIVE 🔮",
        "⭐ DETECTION SPELL CAST ⭐"
    ]

    # Cycle through messages
    msg_index = int(time.time() * 2) % len(messages)
    message = messages[msg_index]

    # Pulsing text effect
    scale = 0.8 + 0.3 * math.sin(time.time() * 4)

    # Add text with glow effect
    cv2.putText(frame, message, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)  # Shadow
    cv2.putText(frame, message, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2)  # Main text


# Open webcam
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None
start_time = None
recording = False

print('Starting YOLOv5 Magic Mirror Detection...')
print('When objects are detected, enjoy the magical effects!')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 expects RGB input
    results = model(frame)

    # Parse results
    detections = results.pandas().xyxy[0]
    detected = False

    for _, row in detections.iterrows():
        label = row['name']
        conf = row['confidence']

        if label in TARGET_CLASSES and conf > 0.5:
            detected = True
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            w, h = x2 - x1, y2 - y1

            # Basic detection rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 4)

            # ✨ MAGIC EFFECTS START HERE ✨

            # Rainbow animated border
            add_rainbow_border(frame, x1, y1, w, h, hash(label) % 100)

            # Sparkle particles
            add_sparkle_effect(frame, x1, y1, w, h)

            # Floating emojis
            add_floating_emojis(frame, x1, y1, w, h, label)

    # Magic mode activation
    if detected:
        if not magic_mode:
            magic_mode = True
            effect_start_time = time.time()

        # Show magic text effects
        add_magic_text_effects(frame)

        # Add screen-wide sparkle effect
        if random.random() < 0.3:  # 30% chance each frame
            for _ in range(3):
                px = random.randint(0, frame.shape[1])
                py = random.randint(0, frame.shape[0])
                color = random.choice([(255, 215, 0), (255, 20, 147), (0, 255, 255)])
                particles.append(Particle(px, py, color))

    # Turn off magic mode after duration
    if magic_mode and time.time() - effect_start_time > EFFECT_DURATION:
        magic_mode = False

    # Always update particles (they fade out naturally)
    particles = [p for p in particles if p.update()]
    for particle in particles:
        particle.draw(frame)

    # Original recording functionality
    if detected and not recording:
        print('🎬 Target class detected. Starting magical recording...')
        start_time = time.time()
        video_writer = cv2.VideoWriter('magical_yolo_detection.mp4', fourcc, 20.0,
                                       (frame.shape[1], frame.shape[0]))
        recording = True

    if recording:
        video_writer.write(frame)
        if time.time() - start_time >= SAVE_DURATION:
            video_writer.release()
            recording = False
            print('✨ Saved 5-second magical video as magical_yolo_detection.mp4')

    # Instructions
    if not detected:
        cv2.putText(frame, 'Show a person, phone, or book for magic! 🪄',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display
    cv2.imshow('YOLOv5 Magic Mirror ✨', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print('Magic mirror session ended! ✨')