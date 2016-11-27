---
layout: post
title: phoneme level rnn
date: 2016-11-27
---

# 본 포스팅의 목적

1. 모듈을 안 쓰고 가장 간단하게 RNN을 짜는 것을 이해한다.
2. 한글을 **자모 단위로 쪼개서** character-level-rnn을 해본다

# 사용될 코드
'Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy' (source: [Karpathy github gist](https://gist.github.com/karpathy/d4dee566867f8291f086#file-min-char-rnn-py-L17))

#### 인촌 서버 상의 저장 위치

	inchon26-worker:/home/hdd2tb/min-rnn$
	*min-rnn 폴더 통째로 가져다 쓰시면 됩니다.


# 코드 플로우

원래는 아래와 같이 같이 numpy 하나만 필요한 스크립트에요.<br/>

```python
import numpy as np
```

텐서 플로우 안 부르냐구요? <br/> 네... 왜냐하면 **이건 텐서플로우 활용 예제가 아니에요.**
**lstm도 안 할겁니다.** <br/>

 가장 기본적인 RNN 코드** 를 100줄로 구현해본 예시라고 합니다. 텐서플로우의 다양한 함수들을 파 보기 전에 간단하게나마
 기본 가닥을 잡고 가는게 좋을것 같아서 살펴보게 되었어요.


지난 포스트에서는 한글 음절 (글자) 단위로 sequence를 예측하는 모델을 만들어 보았으니, 이번에는 **자모 단위로 쪼개서** 한번 해 봅니다. 이를 위해 필요한 모듈인  **hangul_utils** (없을시엔 터미널에서 `pip install hangul_utils`를 실행해 설치)를 가져옵니다.<br/>

```python
from hangul_utils import split_syllables, join_jamos
```
*cf. 비슷한 모듈은 널려 있고 만들기도 어렵지 않으니 꼭 이것을 사용하시지 않아도 됩니다.*

데이터를 다음과 같이 불러왔습니다.

```python
# data I/O
data = open('toji.txt', 'r').read() # should be simple plain text file
```

>data = '웃더니 말했다. "주유가  드디어 죽을 때가 \n가까워진 모양입니다. 제  꾀에 제가 넘어가게 되었으니 어찌 살기를  바랄 수가 \n있겠습니까?...'

자모단위로 쪼개기 위해  `split_syllables` 를 쓸거에요. 다음 라인을 추가해 주었습니다.

```python
data = split_syllables(data)
```

자모 분해가 됐습니다.

>data = 'ㅇㅜㅅㄷㅓㄴㅣ ㅁㅏㄹㅎㅐㅆㄷㅏ. "ㅈㅜㅇㅠㄱㅏ  ㄷㅡㄷㅣㅇㅓ ㅈㅜㄱㅇㅡㄹ ㄸㅐㄱㅏ \nㄱㅏㄲㅏㅇㅝㅈㅣㄴ ㅁㅗㅇㅑㅇㅇㅣㅂㄴㅣㄷㅏ...  '

이제부터는 원 스크립트 그대로 갑니다 .

```python
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

```

seq_length 라는 변수에서 RNN 한 샘플의 크기를 정해주네요.<br/>디폴트 설정으로는 25 character가 한 단위네요. (지금 우리의 경우에는 자음 또는 모음의 나열 25개를 의미하겠군요...)

참고로,  하이퍼 패러미터로 지정하진 않지만, 인풋과 아웃풋의 offset은 1로 설정되어 있어요. 이말인 즉슨,

>ㅇㅜㅅㄷㅓㄴㅣ ㅁㅏㄹㅎㅐㅆㄷ (웃더니 말했ㄷ)

위와 같은 시퀀스를 인풋으로,  아래와 같은 시퀀스를 타겟으로 하는 거죠. 

> ㅜㅅㄷㅓㄴㅣ ㅁㅏㄹㅎㅐㅆㄷㅏ (ㅜㅅ더니말했다)


**character마다 그 다음의 character를 예측하는 모델을 만들려고 하니까 한 칸씩 밉니다.** 이걸 offset=1 이다 라고 간편하게 표현한대요.


```python
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
```
트레이닝 중간중간에 'sampling', 다시 말해  *현재 훈련 상태로는 character sequence 예측력이 얼마나 되는지 확인*해 볼수 있게 되어있네요.<br/>

데이터 100개 (25 글자 * 100개)에 대한 훈련이 끝날때 마다 샘플링을 해서 결과를 보여주나봐요. 

```python
  # sample from the model now and then
  if n % 100 == 0: 
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    jamo_numbers = [char_to_ix[x] for x in txt]
    print(jamo_numbers); 

```

여기에도 약간의 추가 코드를 넣어주어야 해요. 이유인 즉슨, 아까는 쪼개서 분석 했으니, 샘플링 결과도 자모가 분해되어 나오겠죠? 조합해서  보여달라고만 하면 됩니다. 여기서 시작할때 불러온 `join_jamos` 가 사용됩니다. 

```python
    restored_jamo = ''.join([ix_to_char[x] for x in jamo_numbers])
    restored_text = join_jamos(restored_jamo)
    print('----\n %s \n----' % (restored_text, ));
```

나머지 부분은 변경사항 없습니다.

```python

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
```

자 코드를 다 살펴 보았으니... 코드를 돌려봅시다!

	python min-char-rnn.py 
	
결과가 입력창에 계속 뜨네요...  샘플링의 질이 어떻게 달라지는지 살펴보고 다시 오겠습니다.

-youngsunhere