---
layout: post
title: tensorflow install
date: 2016-12-01
---


# 텐서플로 ptbwordlm.py 안 꼬이고 쓰기



1. 텐서플로 깔기


		pip install tensorflow


2. 텐서플로 git repo clone 하기

		git clone https://github.com/tensorflow/tensorflow.git
		
3. 레시피 위치로 가기 **(이때의 tensorflow 폴더는 clone 해온 repo의 하위폴더임!!!** 텐서플로가 *실제*로 깔린 곳에 가면, *필요한 파일들이 제대로 없을 수도...*)
	

		cd tensorflow/models/rnn/ptb
		

4. ptb word_lm.py를 돌리기. 
	
	data path 는 실제 내 데이터의 위치로 변경해줘야 한다.  
	([데이터 받기] (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz); 참고: 링크에서 simple-example 파일을 받은 후에 그 안에 있는 data 디렉토리가 아래의 data path로 지정하삼.)
 
		python ptb_word_lm.py --data_path=/tmp/simple-examples/data/ 	--model small

- 혹시 아래와 같은 에러가 난다면?


		.
		.
		.
		Error importing tensorflow.  Unless you are using bazel,
		you should not try to import tensorflow from its source directory;
		please exit the tensorflow source tree, and relaunch your python interpreter
		from there.
		
 여기저기 뒤져보니, **protobuf** 라는 모듈이 일으키는 문제라고 한다. 해결 방법은 간단하다. 텐서플로 지우고, **protobuf** 지우고, 다시 tensorflow 깐다!
		
		pip uninstall tensorflow
		pip uninstall protobuf
		pip install tensorflow
		

	다 고쳤으면 다시 해본다...
	
		Youngsuns-MacBook-Air-2:ptb yscho$ python ptb_word_lm.py --data_path=/Users/youngsuncho/db/		simple-examples/simple-examples/data --model small
		WARNING:tensorflow:Standard services need a 'logdir' passed to the SessionManager
		Epoch: 1 Learning rate: 1.000
		0.004 perplexity: 5342.714 speed: 805 wps
		0.104 perplexity: 848.289 speed: 650 wps
		0.204 perplexity: 629.971 speed: 596 wps
		0.304 perplexity: 508.013 speed: 663 wps
		0.404 perplexity: 437.958 speed: 678 wps
		0.504 perplexity: 392.499 speed: 645 wps
		0.604 perplexity: 353.477 speed: 646 wps

이제 잘 될 거심!

-youngsunhere
	
	
	