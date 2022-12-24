<h2 align="center">Применение библиотеки Scikit-Learn для решения задач кластеризации</h2>

**Ссылки**:
- [VK - Шныра Богдан ИСТ-19б](https://vk.com/id404101172)


### Инструменты разработки

**Стек:**
- Python >= 3.10
- Pandas >= 1.5.2
- tabulate >= 0.9.0
- scikit-learn >= 1.2.0
- numpy >= 1.23.5


##### 1) Описание предметной области и постановка задачи:

  - [Набор данных Hayes-Roth](http://archive.ics.uci.edu/ml/datasets/Hayes-Roth)

##### 2) Информация о наборе данных:

Эта база данных содержит 5 числовых атрибутов. Во время тестирования используется только подмножество из 3 (последние 3). Кроме того, при тестировании «используются» только 2 из 3 концепций (т.е. с прототипами 000 и 111). Все значения сопоставлены с их эквивалентами с нулевой индексацией.
Некоторые экземпляры могут быть отнесены либо к категории 0, либо к категории 1. 
Фактические значения атрибутов заменены  (например, хобби имеет значения шахматы, спорт и марки) числовыми значениями. 
Информация об атрибутах:
1) имя: отдельное для каждого экземпляра и представленное в числовом виде
2) хобби: номинальные значения в диапазоне от 1 до 3
3) возраст: номинальные значения в диапазоне от 1 до 4
4) уровень образования: номинальные значения в диапазоне от 1 и 4
5) семейное положение: номинальные значения от 1 до 4
6) класс: номинальные значения от 1 до 3.

##### 2) Информация о атрибутах:

1) имя: отдельное для каждого экземпляра и представленное в числовом виде
2) хобби: номинальные значения в диапазоне от 1 до 3
3) возраст: номинальные значения в диапазоне от 1 до 4
4) уровень образования: номинальные значения в диапазоне от 1 и 4
5) семейное положение: номинальные значения от 1 до 4


##### 3) Описание алогоритма:

OPTICS расшифровывается как Ordering points для определения структуры кластеризации. Это алгоритм неконтролируемого обучения на основе плотности, разработанный той же исследовательской группой, что и DBSCAN. У DBSCAN есть главный недостаток, заключающийся в том, что он изо всех сил пытается идентифицировать кластеры в данных различной плотности. Однако OPTICS не требует, чтобы плотность была одинаковой во всем наборе данных.
Подобно DBSCAN, алгоритм OPTICS требует два параметра — параметр ε описывает максимальное расстояние (радиус), принимаемое во внимание, а параметр MinPts описывает число точек, требующихся для образования кластера. Точка p является основной точкой, если по меньшей мере MinPts точек находятся в её ε-окрестности. В отличие от DBSCAN, алгоритм OPTICS рассматривает также точки, которые являются частью более плотного кластера, так что каждой точке назначается основное расстояние, которое описывает расстояние до MinPts-ой ближайшей точки:

изображение 1

Достижимое расстояние точки o от точки p равно либо расстоянию между o и p, либо основному расстоянию точки p, в зависимости от того, какая величина больше:

изображение 2
изображение 3

Используя график достижимости (особый вид древовидной схемы), легко получить иерархическую структуру кластеров. Это двумерный график, на котором точки по оси x откладываются в порядке их обработки алгоритмом OPTICS, а по оси y откладывается достижимое расстояние. Поскольку точки, принадлежащие кластеру, имеют небольшое достижимое расстояние до ближайшего соседа, кластеры выглядят как долины на графике достижимости. Чем глубже долина, тем плотнее кластер.


    
## License

Copyright (c) 2022-present, - Shnyra Bogdan