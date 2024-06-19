# Project - 포물선 운동 탐구

* Motivation
  * 보통 전공과목에서 공부하는 내용의 수학들은 논리를 이해하는 것을 주 목적으로 하기에 펜과 종이만 있으면 해결 가능한 것들이었다.
  * 지금까지 공부한 것들을 시각화하고, 애니메이션을 만들어 봄으로써 수학이론들을 다각적으로 이해할 수 있는 툴들을 만들어 놓는 게 앞으로 공부할 머신러닝, 데이터 분석을 위한 기본 바탕이 될 수 있다고 생각했다.
  * 파이썬을 이제 막 시작한 사람으로서, 내가 잘 알고 있는 수학, 물리이론을 택해서 코드를 짜는 것이 쉽다고 생각했기에 프로젝트 주제로 '포물선 운동 탐구'를 택하게 됐다.
* Goal
  * Project(1) - draw ideal case(no air friction)
  * project(2) - make animation for ideal case
  * project(3) - using 'gradient ascent' vs using 'Newton method' to find maximum height 


## Project(1) - draw ideal case(no air friction)

* matlopolib library를 사용하여 물체의 포물선 운동을 그려보았다.
* 먼저 포물선 그래프의 frame을 잡아주는 parabolic_graph 함수를 정의한다.
```
def parabolic_graph(x,y):
    plt.plot(x,y)
    plt.xlabel('x_coordinate')
    plt.ylabel('y_coordinate')
    plt.title('Parabolic Motion')
```
* 이 부분이 중요한데 체공시간(flight_time)동안 점을 찍어주는 함수를 정의해야 한다.
* start 부분부터 interval만큼 더해서 등간격으로 점을 찍어준다.
```    
def flight_coordinate(start, final, interval):
    numbers = []
    while start < final:
        numbers.append(start)
        start = start + interval 
    return numbers
```
* 포물선에 관한 식을 알고 있다면 다음과 같은 함수를 정의해줄 수 있다.
* x방향과 y방향으로 나누고 x,y를 각각 t시간때의 위치로 정의한다.
```
def parabolic_trajectory(u, theta):
    # u is intial velocity
    theta = math.radians(theta)
    g = 9.80665
    
    # checking time interval
    t_flight = 2*u*math.sin(theta)/g
    intervals = flight_coordinate(0, t_flight, 0.01)
    
    x = []
    y = []
    for t in intervals:
        x.append(u*math.cos(theta)*t)
        y.append(u*math.sin(theta)*t-0.5*g*(t**2))
        
    parabolic_graph(x,y)
```
## Project(2) - make animation for ideal case

* project(1)에서는 정적인 포물선을 그려보았지만 motion이 있는 animation을 만들고 싶었다.
* u(initial velocity)와 theta(initial angle)를 조절하면 원하는 동영상을 얻을 수 있다.
* construction의 핵심코드만 보겠다.

* 다음 new_position 함수는 t에 따라 움직이는 원의 중심을 표현한 것이다.
```
def new_position(k, circle, intervals, u, theta):
    t = intervals[k]
    x = u*math.cos(theta)*t
    y = u*math.sin(theta)*t - 0.5*g*(t**2)
    circle.center = x,y
    
    return circle
```
* fig = plt.gcf() <-- 이 코드는 figure 객체를 가져온다.
  
* animation class에는 FuncAnimation이라는 메소드가 있는데 설명하자면 다음과 같다.
  
  * fargs는 new_position 함수에 전달할 인수들이다. 튜플로 전달되며, new_position 함수의 매개변수가 된다.
  * frame은 애니메이션의 frame수로 intervals의 길이로 정의된다.
  * interval은 5ms로 두었는데, 이는 다음프레임으로 넘어가는 시간으로 보면 된다. 짧으면 짧을 수록 영상 속도가 빨라진다.
```
fig = plt.gcf()
    ax = plt.axes(xlim = (x_min, x_max), ylim = (y_min, y_max))
    circle = plt.Circle((x_min, y_min), 1.0)
    ax.add_patch(circle)
    
    ani = animation.FuncAnimation(fig, new_position, fargs = (circle, intervals, u, theta), frames = len(intervals), interval = 5, repeat = False)
```

## Project(3) - using 'gradient ascent' vs using 'Newton method' to find maximum height 

* 함수의 극소/극댓값을 찾는 여러가지 optimization 방식이 있는데, 매트랩으로 구현했던 과정을 파이썬으로도 구현해보려고 한다.
  
* 대표적으로 'gradient ascent' 방법과 'Newton method'를 이용해보려고 한다.
  
* 이 방식으로 어떤 u(initial veloctiy)가 주어지더라도 수평거리를 길게 만들려면 theta가 45도여야하는 것을 보일 것이다.

### Gradient ascent

* sympy는 symbolic 표현식을 연산하기 위한 모듈로, 여기서는 미분계산을 위해 도입하였다.
  
* time 모듈은 Gradient ascent 모델의 성능을 측정하기 위해서 도입했다.
'''
import math
from sympy import Derivative, Symbol, sin
import time
'''

* gradient ascent 함수는 다음과 같이 정의된다.
  
  * 수치적 방법으로 theta 값을 얻을 것이기 때문에 tolerance 변수를 도입했다.
    
  * alpha는 gradient 방향으로 얼마만큼 움직일 것인지를 표현하는 변수다.
    
  * x_new는 x_old로부터 inductive하게 얻어지는 값이다.
    
  * 이 함수는 x_new와 x_old의 차이가 tolerance보다 작아지면 return을 한다.
```
def gradient_ascent(x_0, flx, x):
    tolerance = 1e-6
    alpha = 1e-4
    x_old = x_0
    x_new = x_old + alpha * flx.subs({x:x_old}).evalf()
    
    while abs(x_old - x_new) > tolerance:
        x_old = x_new
        x_new = x_old + alpha * flx.subs({x:x_old}).evalf()
    
    return x_new
```
* 다음 함수는 get_max_theta 함수로 theta 값의 maximum을 찾아준다.

* 초기 theta 값은 10^(-3)부터 시작했다.

* execution_time은 gradient_ascent 함수의 작동 시간으로 gradient_ascent 함수의 성능지표다.
```
def get_max_theta(R, theta):
    R1theta = Derivative(R, theta).doit()
    start_time = time.time()  # 시작 시간 측정
    theta0 = 1e-3
    theta_max = gradient_ascent(theta0, R1theta, theta)
    end_time = time.time()  # 종료 시간 측정
    execution_time = end_time - start_time
    return theta_max, execution_time
```
#### Conclusion
  * what is the initial value? 25
  * Theta: 44.99785585098667
  * Maximum Range: 63.732263132614
  * Execution Time: 0.278365 seconds

### Newton's Method

* newtons_method 함수는 다음과 같이 정의된다.
  
  * x_new와 x_old의 차이가 tol보다 작아지면 return한다.
 
  * f의 극대/극소를 찾아야하기 때문에 f''(double prime)과 f'이 필요하다.
```
def newtons_method(x_0, f_double_prime, f_prime, x):
    tol = 1e-6
    x_old = x_0
    
    while True:
        f_double_prime_val = f_double_prime.subs({x: x_old}).evalf()
        f_prime_val = f_prime.subs({x: x_old}).evalf()
        x_new = x_old - f_prime_val / f_double_prime_val
        
        if abs(x_new - x_old) < tol:
            break
        x_old = x_new
    
    return x_new
```
* get_max_theta 함수를 정의한다.

  * R1theta는 R을 theta로 미분한 것이다.
    
  * R2theta는 R1theta를 theta로 미분한 것으로, 결과적으로 R을 theta로 두 번 미분한 것이다.

  * execution_time은 newtons_method 함수의 작동 시간으로 gradient_ascent 함수의 성능지표다.
```
def get_max_theta(R, theta):
    R1theta = Derivative(R, theta).doit()
    R2theta = Derivative(R1theta, theta).doit()
    start_time = time.time() 
    theta0 = 45
    theta_max = newtons_method(theta0, R2theta, R1theta, theta)
    end_time = time.time()  
    execution_time = end_time - start_time 
    
    return theta_max, execution_time
```
#### conclusion
  * what is the initial value? 25
    
  * Theta: 2565.000000000000
    
  * Maximum Range: 63.7322633111205
    
  * Execution Time: 0.039384 seconds
    
#### 비정상적인 값이 나왔는데 그 이유는 R이 sin function이라 convex하지 않기 때문이다.
#### newton method는 initial value가 중요한데 수렴값에 멀리 떨어지면 수렴하지 않을 수 있다.
