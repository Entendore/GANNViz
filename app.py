import sys
import json
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, 
                               QButtonGroup, QRadioButton, QFrame, QGroupBox, QCheckBox, 
                               QScrollArea, QPlainTextEdit, QToolBar)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QLineF, QThread, Signal, QTime
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QMouseEvent, QPolygonF, 
                           QKeyEvent, QRadialGradient, QLinearGradient, QFont)

# --- Particle System ---
class Particle:
    def __init__(self, x, y, color):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 3.0
        self.life = 1.0
        self.color = color

    def update(self):
        self.pos += self.vel
        self.vel *= 0.92
        self.life -= 0.04

# --- Dynamic Neural Network ---

class NeuralNetwork:
    def __init__(self, layer_sizes, weights=None):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w_shape = (layer_sizes[i], layer_sizes[i+1])
            b_shape = (layer_sizes[i+1],)
            if weights is None:
                self.weights.append(np.random.randn(*w_shape) * 0.5)
                self.biases.append(np.random.randn(*b_shape) * 0.5)
            else:
                self.weights.append(np.array(weights[i][0]))
                self.biases.append(np.array(weights[i][1]))

    def forward(self, inputs):
        self.activations = [inputs]
        current = inputs
        for w, b in zip(self.weights, self.biases):
            current = np.tanh(np.dot(current, w) + b)
            self.activations.append(current)
        return current

    def get_flat_weights(self):
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    def set_flat_weights(self, flat_weights):
        idx = 0
        new_weights = []
        new_biases = []
        for i in range(len(self.layer_sizes) - 1):
            w_shape = (self.layer_sizes[i], self.layer_sizes[i+1])
            b_shape = (self.layer_sizes[i+1],)
            
            w_size = np.prod(w_shape)
            w = flat_weights[idx:idx+w_size].reshape(w_shape)
            idx += w_size
            
            b_size = np.prod(b_shape)
            b = flat_weights[idx:idx+b_size].reshape(b_shape)
            idx += b_size
            
            new_weights.append(w)
            new_biases.append(b)
            
        self.weights = new_weights
        self.biases = new_biases

    def to_dict(self):
        return {
            'layer_sizes': self.layer_sizes,
            'weights': [[w.tolist() for w in self.weights], [b.tolist() for b in self.biases]]
        }

class Car:
    def __init__(self, start_pos, start_angle, checkpoints, obs_list, num_rays, fov, layer_sizes, color=None, nn_weights=None):
        self.start_pos = np.array(start_pos, dtype=float)
        self.pos = np.array(start_pos, dtype=float)
        self.start_angle = start_angle
        self.angle = start_angle
        self.speed = 0.0
        self.max_speed_achieved = 0.0
        
        self.checkpoints = checkpoints
        self.obs_list = obs_list 
        
        self.num_rays = num_rays
        self.fov = fov
        
        if color is None:
            self.color = np.random.randint(50, 255, 3)
        else:
            self.color = color
        
        self.nn = NeuralNetwork(layer_sizes, nn_weights)
        self.fitness = 0.0
        self.dead = False
        self.just_died = False
        self.step_count = 0
        self.max_steps = 2500 
        self.steps_since_cp = 0 
        
        self.current_checkpoint_idx = 0
        self.total_checkpoints_passed = 0
        self.laps = 0
        
        self.ray_distances = []
        self.ray_endpoints = []
        self.sensor_inputs = []
        self.current_outputs = [0.0, 0.0]
        
        self.trail = [(self.pos.copy(), self.speed)]

    def cast_rays(self):
        max_dist = 150.0
        half_fov = np.deg2rad(self.fov / 2.0)
        angles = np.linspace(-half_fov, half_fov, self.num_rays)
        
        rays = []
        endpoints = []
        
        rects = [o for o in self.obs_list if o[0] == 'r']
        circles = [o for o in self.obs_list if o[0] == 'c']
        
        for angle_offset in angles:
            ray_angle = self.angle + angle_offset
            dx, dy = np.cos(ray_angle), np.sin(ray_angle)
            
            min_dist = max_dist
            end_pos = self.pos + np.array([dx, dy]) * max_dist
            
            if len(rects) > 0:
                obs_arr = np.array(rects)
                ox, oy, ow, oh = obs_arr[:,1], obs_arr[:,2], obs_arr[:,3], obs_arr[:,4]
                
                inv_dx = 1e9 if abs(dx) < 1e-6 else 1.0 / dx
                inv_dy = 1e9 if abs(dy) < 1e-6 else 1.0 / dy
                
                tx1 = (ox - self.pos[0]) * inv_dx
                tx2 = (ox + ow - self.pos[0]) * inv_dx
                ty1 = (oy - self.pos[1]) * inv_dy
                ty2 = (oy + oh - self.pos[1]) * inv_dy
                
                tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
                tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
                
                valid = (tmax >= 0) & (tmin <= tmax) & (tmin >= 0) & (tmin < min_dist)
                if np.any(valid):
                    min_dist = np.min(tmin[valid])
                    end_pos = self.pos + np.array([dx, dy]) * min_dist

            if len(circles) > 0:
                circ_arr = np.array(circles)
                cx, cy, r = circ_arr[:,1], circ_arr[:,2], circ_arr[:,3]
                
                vx = self.pos[0] - cx
                vy = self.pos[1] - cy
                t = -(vx * dx + vy * dy)
                
                valid = t > 0
                if np.any(valid):
                    dist_sq = (vx + t * dx)**2 + (vy + t * dy)**2
                    valid = valid & (dist_sq < r**2)
                    if np.any(valid):
                        thc = np.sqrt(r[valid]**2 - dist_sq[valid])
                        t_hit = t[valid] - thc
                        best_t = np.min(t_hit[t_hit > 0])
                        if best_t < min_dist:
                            min_dist = best_t
                            end_pos = self.pos + np.array([dx, dy]) * min_dist

            if dx > 0:
                t = (800 - self.pos[0]) * inv_dx
                if 0 <= t < min_dist: min_dist, end_pos = t, self.pos + np.array([dx, dy]) * t
            elif dx < 0:
                t = (0 - self.pos[0]) * inv_dx
                if 0 <= t < min_dist: min_dist, end_pos = t, self.pos + np.array([dx, dy]) * t
                    
            if dy > 0:
                t = (600 - self.pos[1]) * inv_dy
                if 0 <= t < min_dist: min_dist, end_pos = t, self.pos + np.array([dx, dy]) * t
            elif dy < 0:
                t = (0 - self.pos[1]) * inv_dy
                if 0 <= t < min_dist: min_dist, end_pos = t, self.pos + np.array([dx, dy]) * t

            rays.append(min_dist / max_dist)
            endpoints.append(end_pos)
            
        self.ray_distances = rays
        self.ray_endpoints = endpoints
        return rays

    def update(self):
        if self.dead:
            return

        self.just_died = False
        rays = self.cast_rays()
        
        target = self.checkpoints[self.current_checkpoint_idx]
        dx_target = target[0] - self.pos[0]
        dy_target = target[1] - self.pos[1]
        
        target_angle = np.arctan2(dy_target, dx_target)
        angle_diff = target_angle - self.angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        norm_angle_diff = angle_diff / np.pi
        
        inputs = np.array([*rays, self.speed / 5.0, norm_angle_diff])
        self.sensor_inputs = inputs
        
        outputs = self.nn.forward(inputs)
        steering = outputs[0] 
        throttle = outputs[1] 
        self.current_outputs = outputs
        
        self.speed += throttle * 0.2
        self.speed = np.clip(self.speed, -1.0, 5.0) 
        self.angle += steering * 0.15 * (max(self.speed, 0.1) / 5.0) 
        
        self.pos[0] += np.cos(self.angle) * self.speed
        self.pos[1] += np.sin(self.angle) * self.speed
        
        if self.speed > self.max_speed_achieved:
            self.max_speed_achieved = self.speed
        
        self.step_count += 1
        self.steps_since_cp += 1

        if self.step_count % 10 == 0:
            self.trail.append((self.pos.copy(), self.speed))
            if len(self.trail) > 150:
                self.trail.pop(0)

        if self.pos[0] < 0 or self.pos[0] > 800 or self.pos[1] < 0 or self.pos[1] > 600:
            self.dead = True
            self.just_died = True
            
        point = QPointF(self.pos[0], self.pos[1])
        for obs in self.obs_list:
            if obs[0] == 'r':
                if QRectF(obs[1], obs[2], obs[3], obs[4]).contains(point):
                    self.dead = True
                    self.just_died = True
                    break
            elif obs[0] == 'c':
                if (self.pos[0]-obs[1])**2 + (self.pos[1]-obs[2])**2 < obs[3]**2:
                    self.dead = True
                    self.just_died = True
                    break
                
        dist_to_cp = np.linalg.norm(self.pos - target)
        if dist_to_cp < 35:
            self.current_checkpoint_idx = (self.current_checkpoint_idx + 1) % len(self.checkpoints)
            self.total_checkpoints_passed += 1
            if self.current_checkpoint_idx == 0:
                self.laps += 1
            self.steps_since_cp = 0 
                
        if self.step_count >= self.max_steps or self.steps_since_cp > 150:
            self.dead = True

    def calculate_fitness(self):
        next_cp = self.checkpoints[self.current_checkpoint_idx]
        dist = np.linalg.norm(self.pos - next_cp)
        self.fitness = (self.total_checkpoints_passed * 20.0) + (1.0 / (dist + 1.0))
        if self.dead and self.total_checkpoints_passed == 0:
            self.fitness *= 0.1 

class GeneticAlgorithm:
    def __init__(self, pop_size, start_pos, start_angle, checkpoints, obs_list, num_rays, fov, hidden_layers, hidden_neurons):
        self.pop_size = pop_size
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.checkpoints = checkpoints
        self.obs_list = obs_list
        
        self.num_rays = num_rays
        self.fov = fov
        self.layer_sizes = [num_rays + 2] + [hidden_neurons] * hidden_layers + [2]
        
        self.population = [Car(start_pos, start_angle, checkpoints, obs_list, num_rays, fov, self.layer_sizes) for _ in range(pop_size)]
        self.generation = 1
        self.mutation_rate = 0.05
        self.auto_mutation = False
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        self.ghost_trail = []
        self.ghost_color = [100, 100, 100]
        
        self.all_time_best_trail = []
        self.all_time_best_fitness = 0.0

    def evolve(self):
        for car in self.population:
            car.calculate_fitness()
            
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_dot = self.population[0]
        avg_fitness = np.mean([c.fitness for c in self.population])
        
        self.best_fitness_history.append(best_dot.fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        if self.auto_mutation and len(self.avg_fitness_history) > 1:
            if avg_fitness < self.avg_fitness_history[-2]:
                self.mutation_rate = min(0.2, self.mutation_rate + 0.01)
            else:
                self.mutation_rate = max(0.01, self.mutation_rate - 0.005)
        
        self.ghost_trail = best_dot.trail.copy()
        self.ghost_color = best_dot.color.copy()
        
        if best_dot.fitness > self.all_time_best_fitness:
            self.all_time_best_fitness = best_dot.fitness
            self.all_time_best_trail = best_dot.trail.copy()
        
        new_population = []
        
        best_weights = best_dot.nn.get_flat_weights()
        elite = Car(self.start_pos, self.start_angle, self.checkpoints, self.obs_list, self.num_rays, self.fov, self.layer_sizes, color=best_dot.color.copy())
        elite.nn.set_flat_weights(best_weights.copy())
        new_population.append(elite)
        
        while len(new_population) < self.pop_size:
            parent1_w, parent1_c = self.select_parent()
            parent2_w, parent2_c = self.select_parent()
            child_weights = self.crossover(parent1_w, parent2_w)
            child_weights, child_color = self.mutate(child_weights, parent1_c, parent2_c)
            
            child = Car(self.start_pos, self.start_angle, self.checkpoints, self.obs_list, self.num_rays, self.fov, self.layer_sizes, color=child_color)
            child.nn.set_flat_weights(child_weights)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1

    def select_parent(self):
        contenders = np.random.choice(self.population, 5)
        contenders = sorted(contenders, key=lambda x: x.fitness, reverse=True)
        return contenders[0].nn.get_flat_weights(), contenders[0].color.copy()

    def crossover(self, p1, p2):
        mask = np.random.rand(len(p1)) < 0.5
        return np.where(mask, p1, p2)

    def mutate(self, weights, p1_color, p2_color):
        mask = np.random.rand(len(weights)) < self.mutation_rate
        noise = np.random.randn(len(weights)) * 0.5
        weights = np.where(mask, weights + noise, weights)
        
        child_color = (p1_color + p2_color) // 2
        color_mut_mask = np.random.rand(3) < 0.1
        child_color[color_mut_mask] += np.random.randint(-30, 30, np.sum(color_mut_mask))
        child_color = np.clip(child_color, 50, 255)
        return weights, child_color

# --- Background Threading for Turbo Mode ---
class TurboThread(QThread):
    gens_run = Signal(int)
    finished = Signal()

    def __init__(self, ga, num_gens):
        super().__init__()
        self.ga = ga
        self.num_gens = num_gens

    def run(self):
        for _ in range(self.num_gens):
            while True:
                all_dead = True
                for car in self.ga.population:
                    car.update()
                    if not car.dead:
                        all_dead = False
                if all_dead:
                    break
            self.ga.evolve()
            self.gens_run.emit(1)
        self.finished.emit()

# --- PySide6 GUI ---

class NetworkVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 350)
        self.nn = None

    def update_network(self, nn):
        self.nn = nn
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(10, 15, "Neural Network (Best Car)")
        
        if not self.nn: return

        layer_sizes = self.nn.layer_sizes
        activations = self.nn.activations if hasattr(self.nn, 'activations') else [np.zeros(s) for s in layer_sizes]
        weights = self.nn.weights
        
        node_positions = []
        width = self.width() - 40
        height = self.height() - 60
        x_step = width / (len(layer_sizes) - 1)

        for i, size in enumerate(layer_sizes):
            x = 20 + i * x_step
            y_step = height / max(size, 1)
            layer_nodes = []
            for j in range(size):
                y = 30 + j * y_step + (y_step / 2)
                layer_nodes.append((x, y))
            node_positions.append(layer_nodes)

        for i in range(len(weights)):
            w = weights[i]
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    val = w[j, k]
                    color = QColor(0, 255, 0, int(abs(val) * 100)) if val > 0 else QColor(255, 0, 0, int(abs(val) * 100))
                    painter.setPen(QPen(color, abs(val) * 2))
                    p1 = node_positions[i][j]
                    p2 = node_positions[i+1][k]
                    painter.drawLine(QLineF(p1[0], p1[1], p2[0], p2[1]))

        painter.setPen(Qt.NoPen)
        for i, layer in enumerate(node_positions):
            for j, (x, y) in enumerate(layer):
                act = activations[i][j] if i < len(activations) and j < len(activations[i]) else 0
                intensity = int(abs(act) * 255)
                color = QColor(100, 100, 255, intensity) if act > 0 else QColor(255, 100, 100, intensity)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x, y), 6, 6)
                painter.setBrush(QBrush(QColor(200, 200, 200)))
                painter.drawEllipse(QPointF(x, y), 2, 2)
                
        painter.setPen(QPen(QColor(150, 150, 150)))
        painter.drawText(5, self.height() - 25, "In: Rays, Spd, Ang")
        painter.drawText(180, self.height() - 25, "Out: Steer, Accel")

class FitnessGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)
        self.best_history = []
        self.avg_history = []
        self.show_best = True
        self.show_avg = True

    def set_history(self, best, avg):
        self.best_history = best
        self.avg_history = avg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if len(self.best_history) < 2:
            return
            
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.drawLine(0, self.height()-10, self.width(), self.height()-10)
        
        max_val = max(self.best_history) if max(self.best_history) > 0 else 1.0
        
        if self.show_avg:
            painter.setPen(QPen(QColor(50, 150, 255), 2))
            points = []
            for i, val in enumerate(self.avg_history):
                x = (i / (len(self.avg_history) - 1)) * self.width()
                y = self.height() - 10 - (val / max_val) * (self.height() - 20)
                points.append(QPointF(x, y))
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i+1])
            
        if self.show_best:
            painter.setPen(QPen(QColor(0, 255, 100), 2))
            points = []
            for i, val in enumerate(self.best_history):
                x = (i / (len(self.best_history) - 1)) * self.width()
                y = self.height() - 10 - (val / max_val) * (self.height() - 20)
                points.append(QPointF(x, y))
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i+1])

class SimulationCanvas(QWidget):
    def __init__(self, ga):
        super().__init__()
        self.ga = ga
        self.setFixedSize(800, 600)
        self.setMouseTracking(True)
        
        self.edit_mode = "Walls"
        self.drawing_obstacle = False
        self.temp_rect = None
        self.start_drag = None
        self.last_brush_pos = None
        
        self.dragging_cp_idx = -1
        self.dragging_start = False

        self.show_sensors = True
        self.show_trails = True
        self.show_ghost = True
        self.high_quality = True

        self.particles = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(30)

    def update_simulation(self):
        steps_per_frame = self.parent().parent().steps_slider.value()
        
        for _ in range(steps_per_frame):
            all_dead = True
            for car in self.ga.population:
                was_dead = car.dead
                car.update()
                if not was_dead and car.just_died:
                    for _ in range(8):
                        self.particles.append(Particle(car.pos[0], car.pos[1], car.color))
                
                if not car.dead:
                    all_dead = False
                    
            if all_dead:
                self.ga.evolve()
                self.parent().parent().update_stats()
                break 
                
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            
        self.update()

    def draw_grid(self, painter):
        painter.setPen(QPen(QColor(35, 35, 35), 1))
        for x in range(0, 800, 40):
            painter.drawLine(x, 0, x, 600)
        for y in range(0, 600, 40):
            painter.drawLine(0, y, 800, y)

    def draw_hud(self, painter):
        # Draw On-Canvas HUD
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(10, 10, 160, 80, 10, 10)
        
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(20, 30, f"Gen: {self.ga.generation}")
        painter.drawText(20, 50, f"Alive: {sum(1 for c in self.ga.population if not c.dead)}")
        
        live_cars = [c for c in self.ga.population if not c.dead]
        if live_cars:
            best_live = max(live_cars, key=lambda x: x.fitness)
            painter.drawText(20, 70, f"Spd: {best_live.speed:.1f}")
            painter.drawText(20, 85, f"Str: {best_live.current_outputs[0]:.2f}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        if self.high_quality:
            self.draw_grid(painter)
            
        self.draw_hud(painter)
        
        # Draw Obstacles
        for obs in self.ga.obs_list:
            if obs[0] == 'r':
                grad = QLinearGradient(obs[1], obs[2], obs[1], obs[2]+obs[4])
                grad.setColorAt(0, QColor(180, 60, 60))
                grad.setColorAt(1, QColor(120, 30, 30))
                painter.setBrush(QBrush(grad))
                painter.setPen(QPen(QColor(200, 80, 80), 1))
                painter.drawRoundedRect(QRectF(obs[1], obs[2], obs[3], obs[4]), 2, 2)
            elif obs[0] == 'c':
                grad = QRadialGradient(obs[1]-obs[3]*0.3, obs[2]-obs[3]*0.3, obs[3]*1.5)
                grad.setColorAt(0, QColor(220, 150, 50))
                grad.setColorAt(1, QColor(160, 80, 20))
                painter.setBrush(QBrush(grad))
                painter.setPen(QPen(QColor(200, 100, 30), 1))
                painter.drawEllipse(QPointF(obs[1], obs[2]), obs[3], obs[3])
            
        if self.drawing_obstacle and self.temp_rect:
            painter.setBrush(QBrush(QColor(150, 50, 50, 100)))
            painter.setPen(QPen(QColor(255, 100, 100), 1, Qt.DashLine))
            painter.drawRect(self.temp_rect)

        # Draw Checkpoints
        time_pulse = (QTime.currentTime().msec() % 1000) / 1000.0
        pulse_size = np.sin(time_pulse * 2 * np.pi) * 5
        
        for i, cp in enumerate(self.ga.checkpoints):
            is_target = (i == 0)
            radius = 35 + (pulse_size if is_target else 0)
            alpha = 80 if is_target else 30
            
            if self.high_quality:
                grad = QRadialGradient(cp[0], cp[1], radius)
                grad.setColorAt(0, QColor(0, 255, 0, alpha))
                grad.setColorAt(1, QColor(0, 255, 0, 0))
                painter.setBrush(QBrush(grad))
            else:
                painter.setBrush(QBrush(QColor(0, 255, 0, alpha)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cp[0], cp[1]), radius, radius)
            
            painter.setPen(QPen(QColor(0, 255, 0, 200), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cp[0], cp[1]), 35, 35)
            
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QRectF(cp[0]-15, cp[1]-15, 30, 30), Qt.AlignCenter, str(i+1))
            
        # Draw Ghosts
        if self.show_ghost:
            if len(self.ga.all_time_best_trail) > 1:
                painter.setPen(QPen(QColor(255, 255, 255, 100), 4))
                for i in range(len(self.ga.all_time_best_trail) - 1):
                    p1 = QPointF(self.ga.all_time_best_trail[i][0][0], self.ga.all_time_best_trail[i][0][1])
                    p2 = QPointF(self.ga.all_time_best_trail[i+1][0][0], self.ga.all_time_best_trail[i+1][0][1])
                    painter.drawLine(QLineF(p1, p2))

            if len(self.ga.ghost_trail) > 1:
                ghost_color = QColor(int(self.ga.ghost_color[0]), int(self.ga.ghost_color[1]), int(self.ga.ghost_color[2]), 80)
                painter.setPen(QPen(ghost_color, 3))
                for i in range(len(self.ga.ghost_trail) - 1):
                    p1 = QPointF(self.ga.ghost_trail[i][0][0], self.ga.ghost_trail[i][0][1])
                    p2 = QPointF(self.ga.ghost_trail[i+1][0][0], self.ga.ghost_trail[i+1][0][1])
                    painter.drawLine(QLineF(p1, p2))

        # Draw Start
        if self.high_quality:
            grad = QRadialGradient(self.ga.start_pos[0], self.ga.start_pos[1], 20)
            grad.setColorAt(0, QColor(0, 150, 255, 150))
            grad.setColorAt(1, QColor(0, 100, 255, 0))
            painter.setBrush(QBrush(grad))
        else:
            painter.setBrush(QBrush(QColor(0, 100, 255, 100)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(self.ga.start_pos[0], self.ga.start_pos[1]), 20, 20)
        
        live_cars = [c for c in self.ga.population if not c.dead]
        best_live = max(live_cars, key=lambda x: x.fitness) if live_cars else None

        # Draw Particles
        for p in self.particles:
            color = QColor(int(p.color[0]), int(p.color[1]), int(p.color[2]), int(p.life * 255))
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(p.pos[0], p.pos[1]), 3, 3)

        # Draw Cars and Trails
        for car in self.ga.population:
            if self.show_trails and len(car.trail) > 1 and not car.dead:
                for i in range(len(car.trail) - 1):
                    p1, s1 = car.trail[i]
                    p2, s2 = car.trail[i+1]
                    hue = int(240 - (max(s1, 0) / 5.0) * 240)
                    trail_color = QColor.fromHsv(hue, 255, 255, 80)
                    painter.setPen(QPen(trail_color, 2))
                    painter.drawLine(QLineF(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])))

            if car is best_live:
                if self.show_sensors:
                    for i, end_pos in enumerate(car.ray_endpoints):
                        if self.high_quality:
                            color_glow = QColor(0, 255, 255, 30) if car.ray_distances[i] >= 0.95 else QColor(255, 0, 0, 50)
                            painter.setPen(QPen(color_glow, 5))
                            painter.drawLine(QLineF(QPointF(car.pos[0], car.pos[1]), QPointF(end_pos[0], end_pos[1])))
                        
                        color_core = QColor(0, 255, 255, 200) if car.ray_distances[i] >= 0.95 else QColor(255, 50, 50, 255)
                        painter.setPen(QPen(color_core, 1))
                        painter.drawLine(QLineF(QPointF(car.pos[0], car.pos[1]), QPointF(end_pos[0], end_pos[1])))
                
                body_color = QColor(255, 215, 0)
            elif car.dead:
                body_color = QColor(80, 80, 80, 100)
            else:
                body_color = QColor(int(car.color[0]), int(car.color[1]), int(car.color[2]))
            
            painter.save()
            painter.translate(car.pos[0], car.pos[1])
            painter.rotate(np.degrees(car.angle))
            
            # Wheels
            painter.setBrush(QBrush(QColor(20, 20, 20)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(-8, -7, 5, 4, 1, 1)
            painter.drawRoundedRect(-8, 3, 5, 4, 1, 1)
            steer_angle = np.degrees(car.current_outputs[0] * 0.5) if car is best_live else 0
            painter.save()
            painter.translate(6, -5.5)
            painter.rotate(steer_angle)
            painter.drawRoundedRect(-2, -2, 5, 4, 1, 1)
            painter.restore()
            painter.save()
            painter.translate(6, 5.5)
            painter.rotate(steer_angle)
            painter.drawRoundedRect(-2, -2, 5, 4, 1, 1)
            painter.restore()
            
            # Body
            painter.setBrush(QBrush(body_color))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRoundedRect(-10, -5, 20, 10, 3, 3)
            
            # Windshield
            painter.setBrush(QBrush(QColor(100, 200, 255, 200)))
            painter.drawRoundedRect(0, -3, 6, 6, 1, 1)
            
            # Headlights
            painter.setBrush(QBrush(QColor(255, 255, 200, 255)))
            painter.drawEllipse(QPointF(9, -3), 1.5, 1.5)
            painter.drawEllipse(QPointF(9, 3), 1.5, 1.5)
            
            # Brake Lights
            if car.current_outputs[1] < 0:
                painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
            else:
                painter.setBrush(QBrush(QColor(150, 0, 0, 200)))
            painter.drawEllipse(QPointF(-9, -3), 1.5, 1.5)
            painter.drawEllipse(QPointF(-9, 3), 1.5, 1.5)
                
            painter.restore()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        if event.button() == Qt.LeftButton:
            if self.edit_mode == "Walls":
                self.drawing_obstacle = True
                self.start_drag = pos
                self.temp_rect = QRectF(self.start_drag, self.start_drag)
            elif self.edit_mode == "Pylons":
                self.ga.obs_list.append(('c', pos.x(), pos.y(), 15))
                self.update()
            elif self.edit_mode == "Brush":
                self.last_brush_pos = pos
                self.ga.obs_list.append(('r', pos.x()-10, pos.y()-10, 20, 20))
                self.update()
            elif self.edit_mode == "Checkpoints":
                for i, cp in enumerate(self.ga.checkpoints):
                    if np.linalg.norm(np.array([pos.x(), pos.y()]) - np.array(cp)) < 35:
                        self.dragging_cp_idx = i
                        return
                self.ga.checkpoints.append((pos.x(), pos.y()))
                self.update()
            elif self.edit_mode == "Start":
                if np.linalg.norm(np.array([pos.x(), pos.y()]) - np.array(self.ga.start_pos)) < 20:
                    self.dragging_start = True
                else:
                    self.ga.start_pos = (pos.x(), pos.y())
                    self.update()
        elif event.button() == Qt.RightButton:
            if self.edit_mode in ["Walls", "Brush", "Pylons"]:
                for obs in reversed(self.ga.obs_list):
                    if obs[0] == 'r' and QRectF(obs[1], obs[2], obs[3], obs[4]).contains(pos):
                        self.ga.obs_list.remove(obs)
                        break
                    elif obs[0] == 'c' and (pos.x()-obs[1])**2 + (pos.y()-obs[2])**2 < obs[3]**2:
                        self.ga.obs_list.remove(obs)
                        break
            elif self.edit_mode == "Checkpoints":
                if self.ga.checkpoints:
                    closest_idx = np.argmin([np.linalg.norm(np.array([pos.x(), pos.y()])-np.array(cp)) for cp in self.ga.checkpoints])
                    if np.linalg.norm(np.array([pos.x(), pos.y()])-np.array(self.ga.checkpoints[closest_idx])) < 35:
                        self.ga.checkpoints.pop(closest_idx)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self.drawing_obstacle:
            self.temp_rect = QRectF(self.start_drag, pos).normalized()
            self.update()
        elif self.edit_mode == "Brush" and event.buttons() & Qt.LeftButton:
            if not self.last_brush_pos or np.linalg.norm(np.array([pos.x(), pos.y()]) - np.array([self.last_brush_pos.x(), self.last_brush_pos.y()])) > 10:
                self.last_brush_pos = pos
                self.ga.obs_list.append(('r', pos.x()-10, pos.y()-10, 20, 20))
                self.update()
        elif self.dragging_cp_idx != -1:
            self.ga.checkpoints[self.dragging_cp_idx] = (pos.x(), pos.y())
            self.update()
        elif self.dragging_start:
            self.ga.start_pos = (pos.x(), pos.y())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.drawing_obstacle:
                self.drawing_obstacle = False
                if self.temp_rect.width() > 10 and self.temp_rect.height() > 10:
                    self.ga.obs_list.append(('r', self.temp_rect.x(), self.temp_rect.y(), self.temp_rect.width(), self.temp_rect.height()))
                self.temp_rect = None
                self.update()
            elif self.dragging_cp_idx != -1:
                self.dragging_cp_idx = -1
            elif self.dragging_start:
                self.dragging_start = False

# --- Qt Style Sheet ---
DARK_THEME_QSS = """
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: Arial;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px 5px 5px 5px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 5px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #3a3a3a;
    border: 1px solid #6a6a6a;
}
QPushButton:pressed {
    background-color: #1c1c1c;
}
QPushButton#danger { background-color: #c0392b; border: 1px solid #e74c3c; color: white; }
QPushButton#danger:hover { background-color: #e74c3c; border: 1px solid #ff6b6b; }
QPushButton#warn { background-color: #d35400; border: 1px solid #e67e22; color: white; }
QPushButton#warn:hover { background-color: #e67e22; border: 1px solid #ff8c1a; }
QPushButton#turbo { background-color: #8e44ad; border: 1px solid #9b59b6; color: white; }
QPushButton#turbo:hover { background-color: #9b59b6; border: 1px solid #b377cc; }

QSlider::groove:horizontal {
    border: 1px solid #4a4a4a;
    height: 6px;
    background: #2d2d2d;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6a6a6a;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #9a9a9a; }

QRadioButton, QCheckBox {
    spacing: 6px;
    padding: 2px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    background: #2d2d2d;
}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {
    background: #3498db;
    border: 1px solid #5dade2;
}
QPlainTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    font-family: Consolas;
    font-size: 9pt;
}
QScrollArea { border: none; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Evolution Simulator v12: Modern GUI")
        self.resize(1180, 850) # Larger window for new UI
        
        self.pop_size = 100
        self.start_pos = (100, 500)
        self.start_angle = -np.pi/4
        self.checkpoints = [(700, 500), (700, 100), (100, 100)]
        self.obs_list = []
        
        self.num_rays = 7
        self.fov = 180
        self.hidden_layers = 1
        self.hidden_neurons = 12
        
        self.ga = GeneticAlgorithm(self.pop_size, self.start_pos, self.start_angle, self.checkpoints, self.obs_list, self.num_rays, self.fov, self.hidden_layers, self.hidden_neurons)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel)
        
        self.canvas = SimulationCanvas(self.ga)
        left_panel.addWidget(self.canvas)
        
        # Graph Controls Toolbar
        graph_toolbar = QToolBar()
        graph_toolbar.setFixedHeight(30)
        self.cb_show_best = QCheckBox("Show Best")
        self.cb_show_best.setChecked(True)
        self.cb_show_best.setToolTip("Toggle the green Best Fitness line on the graph")
        self.cb_show_best.toggled.connect(lambda s: setattr(self.graph, 'show_best', s))
        self.cb_show_avg = QCheckBox("Show Avg")
        self.cb_show_avg.setChecked(True)
        self.cb_show_avg.setToolTip("Toggle the blue Average Fitness line on the graph")
        self.cb_show_avg.toggled.connect(lambda s: setattr(self.graph, 'show_avg', s))
        self.btn_clear_graph = QPushButton("Clear Graph")
        self.btn_clear_graph.setToolTip("Erase the fitness history graph")
        self.btn_clear_graph.clicked.connect(self.clear_graph)
        
        graph_toolbar.addWidget(self.cb_show_best)
        graph_toolbar.addWidget(self.cb_show_avg)
        graph_toolbar.addWidget(self.btn_clear_graph)
        left_panel.addWidget(graph_toolbar)
        
        self.graph = FitnessGraph()
        left_panel.addWidget(self.graph)
        
        # Event Log Console
        left_panel.addWidget(QLabel("Event Log:"))
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(100)
        self.log_console.setStyleSheet("background-color: #111; color: #2ecc71;")
        left_panel.addWidget(self.log_console)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        main_layout.addWidget(scroll)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(control_panel)
        
        # --- Section: Brain & Sensors ---
        brain_group = QGroupBox("Brain & Sensor Architecture")
        brain_layout = QVBoxLayout(brain_group)
        control_layout.addWidget(brain_group)
        
        brain_layout.addWidget(QLabel("Number of Raycasts (Eyes):"))
        self.rays_slider = QSlider(Qt.Horizontal)
        self.rays_slider.setMinimum(3)
        self.rays_slider.setMaximum(15)
        self.rays_slider.setValue(7)
        self.rays_slider.setToolTip("Number of LiDAR rays cast by the car. More rays = better vision but harder to learn.")
        self.rays_slider.valueChanged.connect(self.update_architecture)
        brain_layout.addWidget(self.rays_slider)
        self.rays_label = QLabel("7 Rays")
        brain_layout.addWidget(self.rays_label)

        brain_layout.addWidget(QLabel("Field of View (Degrees):"))
        self.fov_slider = QSlider(Qt.Horizontal)
        self.fov_slider.setMinimum(30)
        self.fov_slider.setMaximum(180)
        self.fov_slider.setValue(180)
        self.fov_slider.setToolTip("Width of the car's vision cone. 180 is panoramic.")
        self.fov_slider.valueChanged.connect(self.update_architecture)
        brain_layout.addWidget(self.fov_slider)
        self.fov_label = QLabel("180°")
        brain_layout.addWidget(self.fov_label)

        brain_layout.addWidget(QLabel("Hidden Layers:"))
        self.layers_slider = QSlider(Qt.Horizontal)
        self.layers_slider.setMinimum(1)
        self.layers_slider.setMaximum(4)
        self.layers_slider.setValue(1)
        self.layers_slider.setToolTip("Depth of the Neural Network. Deeper networks can learn more complex tracks.")
        self.layers_slider.valueChanged.connect(self.update_architecture)
        brain_layout.addWidget(self.layers_slider)
        self.layers_label = QLabel("1 Layer")
        brain_layout.addWidget(self.layers_label)

        brain_layout.addWidget(QLabel("Neurons per Hidden Layer:"))
        self.neurons_slider = QSlider(Qt.Horizontal)
        self.neurons_slider.setMinimum(4)
        self.neurons_slider.setMaximum(32)
        self.neurons_slider.setValue(12)
        self.neurons_slider.setToolTip("Width of the Neural Network. More neurons = more brain capacity.")
        self.neurons_slider.valueChanged.connect(self.update_architecture)
        brain_layout.addWidget(self.neurons_slider)
        self.neurons_label = QLabel("12 Neurons")
        brain_layout.addWidget(self.neurons_label)

        # --- Section: Simulation Stats ---
        sim_group = QGroupBox("Simulation Stats")
        sim_layout = QVBoxLayout(sim_group)
        control_layout.addWidget(sim_group)
        
        self.gen_label = QLabel("Generation: 1")
        self.gen_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sim_layout.addWidget(self.gen_label)
        
        self.alive_label = QLabel("Alive: 0")
        sim_layout.addWidget(self.alive_label)
        self.fitness_label = QLabel("Best Fitness: 0.0")
        sim_layout.addWidget(self.fitness_label)
        self.mut_label = QLabel("Mutation Rate: 5.0%")
        sim_layout.addWidget(self.mut_label)
        
        self.telemetry_label = QLabel("Speed: 0.0 | Steer: 0.0 | Accel: 0.0")
        self.telemetry_label.setStyleSheet("color: #2ecc71; font-family: monospace;")
        sim_layout.addWidget(self.telemetry_label)
        
        self.adv_telemetry_label = QLabel("Max Spd: 0.0 | Laps: 0 | Time: 0s")
        self.adv_telemetry_label.setStyleSheet("color: #f1c40f; font-family: monospace;")
        sim_layout.addWidget(self.adv_telemetry_label)

        # --- Section: Environment Editor ---
        env_group = QGroupBox("Environment Editor (Hotkeys: W,B,P,C,S)")
        env_layout = QVBoxLayout(env_group)
        control_layout.addWidget(env_group)
        
        mode_group = QButtonGroup(self)
        self.rb_walls = QRadioButton("Draw Rect [W]")
        self.rb_walls.setChecked(True)
        self.rb_brush = QRadioButton("Paint Brush [B]")
        self.rb_pylons = QRadioButton("Place Pylons [P]")
        self.rb_cp = QRadioButton("Edit Checkpoints [C]")
        self.rb_start = QRadioButton("Edit Start [S]")
        
        mode_group.addButton(self.rb_walls)
        mode_group.addButton(self.rb_brush)
        mode_group.addButton(self.rb_pylons)
        mode_group.addButton(self.rb_cp)
        mode_group.addButton(self.rb_start)
        
        self.rb_walls.toggled.connect(lambda: setattr(self.canvas, 'edit_mode', 'Walls'))
        self.rb_brush.toggled.connect(lambda: setattr(self.canvas, 'edit_mode', 'Brush'))
        self.rb_pylons.toggled.connect(lambda: setattr(self.canvas, 'edit_mode', 'Pylons'))
        self.rb_cp.toggled.connect(lambda: setattr(self.canvas, 'edit_mode', 'Checkpoints'))
        self.rb_start.toggled.connect(lambda: setattr(self.canvas, 'edit_mode', 'Start'))
        
        env_layout.addWidget(self.rb_walls)
        env_layout.addWidget(self.rb_brush)
        env_layout.addWidget(self.rb_pylons)
        env_layout.addWidget(self.rb_cp)
        env_layout.addWidget(self.rb_start)

        track_btn_layout = QHBoxLayout()
        self.track_oval_btn = QPushButton("Oval")
        self.track_fig8_btn = QPushButton("Figure-8")
        self.track_maze_btn = QPushButton("Maze")
        self.clear_track_btn = QPushButton("Clear")
        
        self.track_oval_btn.clicked.connect(lambda: self.load_default_track("oval"))
        self.track_fig8_btn.clicked.connect(lambda: self.load_default_track("fig8"))
        self.track_maze_btn.clicked.connect(lambda: self.load_default_track("maze"))
        self.clear_track_btn.clicked.connect(self.clear_track)
        
        track_btn_layout.addWidget(self.track_oval_btn)
        track_btn_layout.addWidget(self.track_fig8_btn)
        track_btn_layout.addWidget(self.track_maze_btn)
        track_btn_layout.addWidget(self.clear_track_btn)
        env_layout.addLayout(track_btn_layout)
        
        self.save_env_btn = QPushButton("Save Track")
        self.save_env_btn.clicked.connect(self.save_env)
        env_layout.addWidget(self.save_env_btn)
        
        self.load_env_btn = QPushButton("Load Track")
        self.load_env_btn.clicked.connect(self.load_env)
        env_layout.addWidget(self.load_env_btn)

        # --- Section: Genetics & Visuals ---
        gen_group = QGroupBox("Genetics & Visuals")
        gen_layout = QVBoxLayout(gen_group)
        control_layout.addWidget(gen_group)

        self.hq_toggle = QCheckBox("High-Quality Graphics (Glows/Grid)")
        self.hq_toggle.setChecked(True)
        self.hq_toggle.setToolTip("Toggles shadows, glowing elements, and grid background for lower-end PCs.")
        self.hq_toggle.toggled.connect(lambda state: setattr(self.canvas, 'high_quality', state))
        gen_layout.addWidget(self.hq_toggle)

        self.sensor_toggle = QCheckBox("Show Best Sensors")
        self.sensor_toggle.setChecked(True)
        self.sensor_toggle.setToolTip("Draws the LiDAR rays of the best currently living car.")
        self.sensor_toggle.toggled.connect(lambda state: setattr(self.canvas, 'show_sensors', state))
        gen_layout.addWidget(self.sensor_toggle)
        
        self.trail_toggle = QCheckBox("Show Speed Trails")
        self.trail_toggle.setChecked(True)
        self.trail_toggle.setToolTip("Draws a heat-map trail behind the cars based on their speed.")
        self.trail_toggle.toggled.connect(lambda state: setattr(self.canvas, 'show_trails', state))
        gen_layout.addWidget(self.trail_toggle)

        self.ghost_toggle = QCheckBox("Show Ghosts")
        self.ghost_toggle.setChecked(True)
        self.ghost_toggle.setToolTip("Draws the path of the previous generation and the all-time best car.")
        self.ghost_toggle.toggled.connect(lambda state: setattr(self.canvas, 'show_ghost', state))
        gen_layout.addWidget(self.ghost_toggle)

        self.auto_mut_toggle = QCheckBox("Auto-Adjust Mutation")
        self.auto_mut_toggle.setToolTip("Dynamically increases mutation if the population stagnates, and decreases it if it's improving.")
        self.auto_mut_toggle.toggled.connect(self.toggle_auto_mut)
        gen_layout.addWidget(self.auto_mut_toggle)
        
        gen_layout.addWidget(QLabel("Manual Mutation Rate:"))
        self.mutation_slider = QSlider(Qt.Horizontal)
        self.mutation_slider.setMinimum(1)
        self.mutation_slider.setMaximum(20)
        self.mutation_slider.setValue(5)
        self.mutation_slider.setToolTip("Chance that a weight in the neural network is randomly altered.")
        self.mutation_slider.valueChanged.connect(self.update_mutation)
        gen_layout.addWidget(self.mutation_slider)

        gen_layout.addWidget(QLabel("Population Size:"))
        self.pop_slider = QSlider(Qt.Horizontal)
        self.pop_slider.setMinimum(10)
        self.pop_slider.setMaximum(250)
        self.pop_slider.setValue(100)
        self.pop_slider.setToolTip("Number of cars spawned per generation. More cars = slower but faster learning.")
        self.pop_slider.valueChanged.connect(self.update_pop_size)
        gen_layout.addWidget(self.pop_slider)

        gen_layout.addWidget(QLabel("Simulation Speed (Steps/Frame):"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(10)
        self.steps_slider.setValue(1)
        self.steps_slider.setToolTip("How fast the physics engine runs. 10x runs 10 seconds in 1 second.")
        gen_layout.addWidget(self.steps_slider)
        
        sim_control_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause [Space]")
        self.pause_btn.setObjectName("warn")
        self.pause_btn.setToolTip("Pause/Resume the simulation (Spacebar).")
        self.pause_btn.clicked.connect(self.toggle_pause)
        sim_control_layout.addWidget(self.pause_btn)
        
        self.turbo_btn = QPushButton("Turbo (50 Gens)")
        self.turbo_btn.setObjectName("turbo")
        self.turbo_btn.setToolTip("Runs 50 generations in a background thread at maximum CPU speed.")
        self.turbo_btn.clicked.connect(self.run_turbo)
        sim_control_layout.addWidget(self.turbo_btn)
        gen_layout.addLayout(sim_control_layout)

        brain_btn_layout = QHBoxLayout()
        self.save_brain_btn = QPushButton("Save Brain")
        self.load_brain_btn = QPushButton("Load Brain")
        self.save_brain_btn.setToolTip("Export the best neural network to a JSON file.")
        self.load_brain_btn.setToolTip("Import a previously saved neural network and apply it to all cars.")
        self.save_brain_btn.clicked.connect(self.save_brain)
        self.load_brain_btn.clicked.connect(self.load_brain)
        brain_btn_layout.addWidget(self.save_brain_btn)
        brain_btn_layout.addWidget(self.load_brain_btn)
        gen_layout.addLayout(brain_btn_layout)

        self.reset_btn = QPushButton("Reset Evolution")
        self.reset_btn.setObjectName("danger")
        self.reset_btn.setToolTip("Wipe all evolutionary progress and start from Generation 1.")
        self.reset_btn.clicked.connect(self.reset_cars)
        gen_layout.addWidget(self.reset_btn)
        
        self.nn_viz = NetworkVisualizer()
        control_layout.addWidget(self.nn_viz)
        
        self.viz_timer = QTimer(self)
        self.viz_timer.timeout.connect(self.update_viz)
        self.viz_timer.start(50)

        self.update_stats()
        self.setFocusPolicy(Qt.StrongFocus)
        self.log("Neuro-Evolution Simulator initialized. Ready to evolve!")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {message}")

    def clear_graph(self):
        self.ga.best_fitness_history = []
        self.ga.avg_fitness_history = []
        self.update_stats()
        self.log("Graph history cleared.")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key_W:
            self.rb_walls.setChecked(True)
        elif event.key() == Qt.Key_B:
            self.rb_brush.setChecked(True)
        elif event.key() == Qt.Key_P:
            self.rb_pylons.setChecked(True)
        elif event.key() == Qt.Key_C:
            self.rb_cp.setChecked(True)
        elif event.key() == Qt.Key_S:
            self.rb_start.setChecked(True)

    def toggle_pause(self):
        if self.canvas.timer.isActive():
            self.canvas.timer.stop()
            self.pause_btn.setText("Play [Space]")
            self.log("Simulation paused.")
        else:
            self.canvas.timer.start(30)
            self.pause_btn.setText("Pause [Space]")
            self.log("Simulation resumed.")

    def update_architecture(self):
        self.num_rays = self.rays_slider.value()
        self.fov = self.fov_slider.value()
        self.hidden_layers = self.layers_slider.value()
        self.hidden_neurons = self.neurons_slider.value()
        
        self.rays_label.setText(f"{self.num_rays} Rays")
        self.fov_label.setText(f"{self.fov}°")
        self.layers_label.setText(f"{self.hidden_layers} Layer(s)")
        self.neurons_label.setText(f"{self.hidden_neurons} Neurons")
        self.reset_cars()

    def toggle_auto_mut(self, state):
        self.ga.auto_mutation = state
        self.mutation_slider.setEnabled(not state)
        self.log(f"Auto-mutation {'enabled' if state else 'disabled'}.")

    def update_mutation(self, value):
        if not self.ga.auto_mutation:
            self.ga.mutation_rate = value / 100.0

    def update_pop_size(self, value):
        self.pop_size = value
        self.reset_cars()

    def clear_track(self):
        self.ga.obs_list = []
        self.ga.checkpoints = []
        self.reset_cars()
        self.log("Track cleared.")

    def run_turbo(self):
        self.turbo_btn.setEnabled(False)
        self.canvas.timer.stop()
        self.log("Starting Turbo Mode (50 generations)...")
        self.turbo_thread = TurboThread(self.ga, 50)
        self.turbo_thread.gens_run.connect(lambda _: self.update_stats())
        self.turbo_thread.finished.connect(self.finish_turbo)
        self.turbo_thread.start()

    def finish_turbo(self):
        self.turbo_btn.setEnabled(True)
        self.canvas.timer.start(30)
        self.pause_btn.setText("Pause [Space]")
        self.update_stats()
        self.log("Turbo Mode finished. 50 generations evolved.")

    def save_brain(self):
        best_dot = max(self.ga.population, key=lambda x: x.fitness)
        data = best_dot.nn.to_dict()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Brain", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(data, f)
            self.log(f"Brain saved to {file_path}.")

    def load_brain(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Brain", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            weights = data['weights']
            for car in self.ga.population:
                car.nn = NeuralNetwork(data['layer_sizes'], weights)
            self.reset_cars()
            self.log(f"Brain loaded from {file_path}.")

    def load_default_track(self, track_type):
        if track_type == "oval":
            self.ga.obs_list = [('r', 50, 50, 700, 20), ('r', 50, 530, 700, 20), ('r', 50, 50, 20, 500), ('r', 730, 50, 20, 500),
                                ('r', 250, 200, 300, 20), ('r', 250, 380, 300, 20), ('r', 250, 200, 20, 200), ('r', 530, 200, 20, 200)]
            self.ga.checkpoints = [(400, 100), (700, 300), (400, 500), (100, 300)]
            self.ga.start_pos = (400, 150)
            self.ga.start_angle = 0 
        elif track_type == "fig8":
            self.ga.obs_list = [('r', 50, 50, 700, 20), ('r', 50, 530, 700, 20), ('r', 50, 50, 20, 500), ('r', 730, 50, 20, 500),
                                ('r', 200, 200, 400, 20), ('r', 200, 380, 400, 20),
                                ('r', 200, 200, 20, 200), ('r', 580, 200, 20, 200)]
            self.ga.checkpoints = [(400, 100), (700, 150), (700, 450), (400, 500), (100, 450), (100, 150)]
            self.ga.start_pos = (400, 150)
            self.ga.start_angle = 0
        elif track_type == "maze":
            self.ga.obs_list = [
                ('r', 100, 100, 20, 300), ('r', 100, 100, 300, 20), 
                ('r', 200, 200, 20, 300), ('r', 200, 480, 300, 20),
                ('r', 300, 300, 20, 200), ('r', 300, 300, 300, 20),
                ('r', 400, 100, 20, 250), ('r', 400, 100, 300, 20),
                ('r', 500, 200, 20, 300), ('r', 500, 480, 200, 20),
                ('r', 600, 200, 20, 300)
            ]
            self.ga.obs_list.append(('c', 150, 150, 15))
            self.ga.obs_list.append(('c', 450, 400, 15))
            self.ga.checkpoints = [(150, 450), (250, 350), (350, 250), (450, 150), (650, 100), (650, 500)]
            self.ga.start_pos = (50, 550)
            self.ga.start_angle = -np.pi/4
        
        self.log(f"Loaded default track: {track_type}.")
        self.reset_cars()

    def save_env(self):
        data = {
            'start_pos': list(self.ga.start_pos),
            'start_angle': self.ga.start_angle,
            'checkpoints': [list(cp) for cp in self.ga.checkpoints],
            'obstacles': [list(obs) for obs in self.ga.obs_list]
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Environment", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(data, f)
            self.log(f"Environment saved to {file_path}.")

    def load_env(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Environment", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.ga.start_pos = tuple(data['start_pos'])
            self.ga.start_angle = data['start_angle']
            self.ga.checkpoints = [tuple(cp) for cp in data['checkpoints']]
            self.ga.obs_list = [tuple(obs) for obs in data['obstacles']]
            self.reset_cars()
            self.log(f"Environment loaded from {file_path}.")

    def reset_cars(self):
        self.ga.num_rays = self.num_rays
        self.ga.fov = self.fov
        self.ga.layer_sizes = [self.num_rays + 2] + [self.hidden_neurons] * self.hidden_layers + [2]
        
        self.ga.pop_size = self.pop_size
        self.ga.population = [Car(self.ga.start_pos, self.ga.start_angle, self.ga.checkpoints, self.ga.obs_list, self.ga.num_rays, self.ga.fov, self.ga.layer_sizes) for _ in range(self.pop_size)]
        self.ga.generation = 1
        self.ga.best_fitness_history = []
        self.ga.avg_fitness_history = []
        self.ga.ghost_trail = []
        self.ga.all_time_best_trail = []
        self.ga.all_time_best_fitness = 0.0
        self.canvas.particles = []
        self.update_stats()

    def update_viz(self):
        live_cars = [c for c in self.ga.population if not c.dead]
        if live_cars:
            best_live = max(live_cars, key=lambda x: x.fitness)
            self.nn_viz.update_network(best_live.nn)
            
            spd = best_live.speed
            steer = best_live.current_outputs[0]
            accel = best_live.current_outputs[1]
            self.telemetry_label.setText(f"Speed: {spd: .1f} | Steer: {steer: .2f} | Accel: {accel: .2f}")
            
            max_spd = best_live.max_speed_achieved
            laps = best_live.laps
            time_alive = best_live.step_count * (1/30) # Approx 30fps physics
            self.adv_telemetry_label.setText(f"Max Spd: {max_spd: .1f} | Laps: {laps} | Time: {time_alive:.1f}s")

    def update_stats(self):
        self.gen_label.setText(f"Generation: {self.ga.generation}")
        alive_count = sum(1 for c in self.ga.population if not c.dead)
        self.alive_label.setText(f"Alive: {alive_count}")
        best_fitness = max(c.fitness for c in self.ga.population) if self.ga.population else 0.0
        self.fitness_label.setText(f"Best Fitness: {best_fitness:.4f}")
        self.mut_label.setText(f"Mutation Rate: {self.ga.mutation_rate*100:.1f}%")
        self.graph.set_history(self.ga.best_fitness_history, self.ga.avg_fitness_history)
        
        # Log generation completion
        if self.ga.generation > 1 and len(self.ga.best_fitness_history) == self.ga.generation - 1:
            self.log(f"Gen {self.ga.generation-1} evolved. Best Fitness: {best_fitness:.4f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME_QSS) # Apply global dark theme
    window = MainWindow()
    window.show()
    sys.exit(app.exec())