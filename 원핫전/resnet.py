'''
https://bskyvision.com/644
https://lv99.tistory.com/21
https://gaussian37.github.io/dl-concept-resnet/

ResNet이해하기

-> 기존 신경망 : x값을 타겟값y로 매핑하기 위한 H(x)함수를 얻는 것이 목표
-> ResNet 신경망: 
    - H(x) = F(x) + *x 
        *입력값을 출력값에 더해줄수 있는 지름길을 추가 
         WHY? 망이 깊어질수록 성능이 좋다?NO
         망을 깊이 만드는 효과를 만들자
    - H(x) = F(x) + x  -> x는 변할 수 없는 값이므로 F(x)를 0으로 만드는 것(최소화)이 목표
      F(x) = *(H(x) - x) *잔차(Residual)을 최소화하는 것이 목표 

-> VGG19모델을 기본으로 Convolution층을 추가해 깊게 만든 후 지름길(skip connection) 추가 

-> 기존의 레이어가 깊어질수록 발생하는 Vanishing gradient 문제를 개선
-> 인풋x의 값을 그대로 전달 받기 때문에 입력의 작은 변화도 깊은 레이어에서 알아챌 수 있음

-> Residual구조를 변형한 것이 *Bottleneck구조, *Identity Mapping구조
-> Residual구조에서 *Bottleneck구조로 변형하여 Dimension의 Reduction과 Expansion 효과를 주어 연산 시간 감소 성능
    *Bottleneck 구조 
        - Residual 구조에서 1x1 → 3x3(원래 하고자 했던 연산) → 1x1 구조를 이용
        - 차원이 축소된 정보로 연산량을 크게 줄일 수 있음
        - 입력받는 것에 비해 적은 수의 차원으로 채널의 수로, 
        filter의 수로 만들어 준다면- 차원이 축소된 정보로 
        연산량을 크게 줄일 수 있다. 한번 이렇게 줄여 놓으면 뒤로가면 
        연계되는 연산량의 수, 파라미터의 수가 확 줄어들기 때문에 같은 
        컴퓨팅 자원과 시간 자원으로 더 깊은 네트워크를 설계하고 학습할 수 있게 된다. 
        그런데, 차원을 단순히/무작정 작게만 만든다고 다 되는 것은 아니다. 
        적당한 크기가 필요하고, 그 다음의 레이어에서 학습할 만큼은 남겨둔 적당한 차원이어야한다. 
        이러한 구조를 잘 활용한 것이 bottleneck 이라는 구조
    
    *Identity Mapping구조 (구조를 검색해서 보는게 이해가 빠름)
        - Residual 구조가 조금 수정된 Identity Mapping
        - Residual 구조에는 한 단위의 feature map을 추출하고 난 후에 activation function을 적용하는 것이 상식
        - Identity Mapping구조에서는 네트워크 출력값과 identity를 더할 때 activation function을 적용하지 않고 그냥 더하는 구조
        - 정리하면 Conv → BN → ReLU → Conv → BN → ReLU에서 BN → ReLU → Conv  → BN → ReLU → Conv  구조로 변경
        - 후자에서는 skip connection에 어떤 추가 연산도 없이 말 그대로 Gradient Highway가 형성되어 극적인 효과
        - 기존의 ResNet보다 에러율도 더 낮아지고 학습도 더 잘되는 것으로 논문에서 나타남
'''

