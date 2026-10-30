import time
import threading
import pyautogui
import pydirectinput
import cv2
import numpy as np
from ultralytics import YOLO
from pynput import keyboard

# --- НАСТРОЙКИ НЕЙРОСЕТИ ---
MODEL_PATH = 'runs/detect/cookie_detector-7/weights/bestV2.pt'
COOKIE_CLASS = 'game'       
AD_CLASS = 'advertising'    

CONF = {'cookie': 0.4, 'ad': 0.25} 
AD_CLICK_CORNER = False      # False = клик в центр крестика

# --- НАСТРОЙКИ АПГРЕЙДОВ ---
START_X = 1345        
START_Y = 366         
ROW_HEIGHT = 73       
NUM_ROWS = 5          
UPGRADE_EVERY = 20.0  

# --- НАСТРОЙКИ СКОРОСТИ ПЕЧЕНЬКИ ---
COOKIE_CLICKS_PER_SECOND = 80  # 80 кликов в секунду (человек не может, браузер не виснет)

# --- ИНИЦИАЛИЗАЦИЯ ---
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
pydirectinput.PAUSE = 0.0

print("Загрузка модели...")
model = YOLO(MODEL_PATH)

_s = pyautogui.screenshot()
SCALE = _s.width / pyautogui.size().width
print(f'Масштаб экрана: {SCALE}')

lock = threading.Lock()
screen_map = {'cookie': [], 'ad': []}
stop_flag = False
is_running = True 

def on_press(key):
    global is_running
    if key == keyboard.Key.space:
        is_running = not is_running
        print(f"\n{'[▶] ПРОДОЛЖИЛ' if is_running else '[⏸] ПАУЗА'}")

listener = keyboard.Listener(on_press=on_press)
listener.start()

def detect_all():
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    results = model(frame, conf=min(CONF.values()), verbose=False)
    
    found = {'cookie': [], 'ad': []}
    for r in results:
        for box in r.boxes:
            name = model.names[int(box.cls[0])]
            key = {COOKIE_CLASS: 'cookie', AD_CLASS: 'ad'}.get(name)
            
            if key is None:
                continue
                
            conf = float(box.conf[0])
            if conf < CONF[key]:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            found[key].append((
                conf,
                (x1 + x2) / 2 / SCALE,
                (y1 + y2) / 2 / SCALE,
                (x2 - x1) / SCALE,
                (y2 - y1) / SCALE
            ))
    return found

def observer():
    while not stop_flag:
        try:
            f = detect_all()
            with lock:
                screen_map.update(f)
        except Exception:
            pass
        time.sleep(0.1)

def snapshot():
    with lock:
        return {k: list(v) for k, v in screen_map.items()}

def close_ads():
    print("\n[!] Реклама! Кликаю...")
    f = detect_all()
    if not f['ad']:
        return
        
    conf, cx, cy, w, h = max(f['ad'], key=lambda d: d[0])
    
    if AD_CLICK_CORNER:
        x, y = cx + w / 2 - max(6, w * 0.04), cy - h / 2 + max(6, h * 0.04)
    else:
        x, y = cx, cy
        
    # Наводимся, ждем и жмем
    pydirectinput.moveTo(int(x), int(y))
    time.sleep(0.1)
    pydirectinput.click()

def click_upgrades():
    print(f'\n[+] Прошло 20 секунд. Кликаю магазин...')
    for i in range(NUM_ROWS):
        current_y = START_Y + (i * ROW_HEIGHT)
        pydirectinput.moveTo(int(START_X), int(current_y))
        pydirectinput.click()
        time.sleep(0.05) 

if __name__ == '__main__':
    print('\nБот запущен!')
    print('>>> НАЖМИТЕ ПРОБЕЛ, ЧТОБЫ ПОСТАВИТЬ НА ПАУЗУ <<<')
    
    t = threading.Thread(target=observer, daemon=True)
    t.start()
    
    last_upgrade = time.time()
    # Рассчитываем задержку между кликами, чтобы получить нужное количество в секунду
    click_delay = 1.0 / COOKIE_CLICKS_PER_SECOND 
    
    try:
        while True:
            # 1. РЕКЛАМА (высший приоритет)
            if snapshot()['ad']:
                close_ads()
                continue

            # 2. ПАУЗА (пробел)
            if not is_running:
                time.sleep(0.05)
                continue

            # 3. АПГРЕЙДЫ (раз в 20 сек)
            if time.time() - last_upgrade >= UPGRADE_EVERY:
                click_upgrades()
                last_upgrade = time.time() 
                continue

            # 4. ПЕЧЕНЬКА (стабильный поток)
            snap = snapshot()
            if snap['cookie']:
                conf, cx, cy, w, h = max(snap['cookie'], key=lambda d: d[0])
                
                # Наводим курсор ОДИН раз
                pydirectinput.moveTo(int(cx), int(cy))
                
                # Кликаем примерно 1 секунду (например, 80 раз)
                # Затем возвращаемся к проверке рекламы
                for _ in range(COOKIE_CLICKS_PER_SECOND):
                    if not is_running or snapshot()['ad']:
                        break
                    pydirectinput.click()
                    time.sleep(click_delay) # Задержка 12.5 мс (браузер успевает)
                    
            else:
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        listener.stop()
        print('\nОстановлено.')