# Centipede AI Agent 🎮🧠

Trabajo Práctico 2 de la materia Inteligencia Artificial  
Universidad de San Andrés – 2023

## 📌 Descripción

Este proyecto consiste en el desarrollo de un agente inteligente capaz de jugar al clásico videojuego **Centipede** de Atari. El agente observa el entorno en forma de imágenes y toma decisiones en tiempo real para maximizar su puntaje, enfrentando enemigos como ciempiés, arañas y pulgas.

Se trabaja en dos etapas principales:

1. **Checkpoint 0**: Extracción del estado del juego a partir de observaciones visuales.
2. **Entrega final**: Implementación del agente inteligente utilizando técnicas de aprendizaje automático.

---

## 🚀 Objetivo

Construir un agente que supere significativamente el rendimiento de un agente aleatorio, utilizando técnicas de:

- Aprendizaje por refuerzo
- Aprendizaje supervisado
- Aprendizaje no supervisado

El enfoque queda a elección del estudiante, siempre que el comportamiento del agente sea razonablemente inteligente y estratégico.

---

## 🧠 Estructura esperada
agents/

├── turing/

│ ├── agent.py # Clase TuringAgent con lógica del agente

│ └── state_extractor.py # Clase StateExtractor para interpretar observaciones


### Ejemplo de uso:

```python
import gymnasium as gym
from agents.turing.agent import TuringAgent
from agents.turing.state_extractor import StateExtractor

agent = TuringAgent()
extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')
obs, _ = env.reset()

while True:
    state = extractor.extract(obs)
    action = agent.action(state)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(agent.name())
# → {nombre:'Alan', apellido:'Turing', legajo:123456}
