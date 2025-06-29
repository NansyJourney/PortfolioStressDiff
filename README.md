# PortfolioStressDiff
Стресс-тестирование торгового портфеля с использование глубоких генеративных моделей

## Структура проекта
```
PortfolioStressDiff/
│
├── data/                                     # данные проекта  
│   ├── df_final_shorted.xlsx                 # Акции российских компаний
│   ├── my_index_shorted.xlsx                 # Веса в индексе
│   ├── saved_data.csv                        # Российские акции и индекс
│   ├── univariate_returns.npy                # Сгенерированные одномерные доходности
│   ├── univariate_returns_stress_1.npy       # Сгенерированные одномерные доходности при первом стресс-сценарии
│   ├── univariate_returns_stress_2.npy       # Сгенерированные одномерные доходности при втором стресс-сценарии
│   ├── univariate_returns_stress_3.npy       # Сгенерированные одномерные доходности при третьем стресс-сценарии
│   ├── multivariate_returns.npy              # Сгенерированные многомерные доходности
│   ├── multivariate_returns_stress_1.npy     # Сгенерированные многомерные доходности при первом стресс-сценарии
│   ├── multivariate_returns_stress_2.npy     # Сгенерированные многомерные доходности при втором стресс-сценарии
│   ├── multivariate_returns_stress_3.npy     # Сгенерированные многомерные доходности при третьем стресс-сценарии
│
│
├── var_dgm/                                  # Модели для генеративного оценивания рисков (TimeGrad)  
│   ├── basic_models/                         # Базовые модели оценки риска  
│       ├── hist_sim.py                       # Расчет VaR и ES при одномерном моделировании  
│   ├── TimeGrad/                             # Реализация диффузионной модели TimeGrad  
│       ├── epsilon_theta.py                  # Модуль предсказания шума  
│       ├── model.py                          # Архитектура модели  
│   ├── multivariate.ipynb                    # Эксперименты с многомерными данными  
│   ├── univariate.ipynb                      # Эксперименты с одномерными данными  
│ 
│ 
├── Предварительный_анализ_данных_и_моделирование_данных.ipynb
├── Стресс_тестирование.ipynb 
