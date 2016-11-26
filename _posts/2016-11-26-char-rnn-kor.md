---
layout: post
title: char-rnn-kor
date: 2016-11-26
---

<br/><br/><br/>

character-level RNN 모델링 예시

코드 출처
-

Stanford의 Andrej Karpathy 블로그에서 간단한 demo인데, 해보기 쉬워서 공유합니다.


(1) [Karpathy's char-level rnn code]: (https://github.com/karpathy/char-rnn) <br/>


(2) [**제가 돌려본 버젼**]: (https://github.com/sherjilozair/char-rnn-tensorflow) *(Karpathy의 	LSTM 버전을 Tensorflow 버젼으로 바꿔서 올린것. 원래는 Torch로 작성)*

***(2)를 fork 하고, inchon26 서버에 가져온 다음에 실행했어요.***


<br/>해보기
-

저자의 예시에는 shakespeare의 아주 작은 코퍼스 (1MB)가 들어있어요. <br/>한국어로도 해보고 싶어서 갖고 있던 박경리 <토지> 1-2권 텍스트를 긁어 넣어봤어요. <br/>역시 잘 돌아가네요.

	
		텍스트   : 박경리 (토지) 1~2부 (전체 5부)
		처리단위  : 글자
		분량 	: 총 998,862 자



실행하는법:

	cd char_rnn_tensorflow
	python2 train.py
	
	   *만약 저처럼, 다른 데이터 (ex 토지)로 돌리려면,
	    1) data/새폴더명/input.txt 이렇게 넣으시고 
	   	2) python2 train.py --data_dir data/새폴더명  이렇게 실행하세요.
		3) 한글은 utf8 포맷인것 확인하고 돌려야 합니다.


정상적인 실행:

	youngsunhere@inchon26-worker:~/char-rnn-tensorflow$ python2 train.py --data_dir 	data/toji12
	I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
	I tensorflow/stream_executor/dso_loader.cc:105] Couldn't open CUDA library libcudnn.so. LD_LIBRARY_PATH:
	I tensorflow/stream_executor/cuda/cuda_dnn.cc:3448] Unable to load cuDNN DSO
	I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
	I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
	I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
	loading preprocessed files
	I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
	I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties:
	name: GeForce GTX 1080
	major: 6 minor: 1 memoryClockRate (GHz) 1.8475
	pciBusID 0000:01:00.0
	Total memory: 7.92GiB
	Free memory: 7.81GiB
	I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0
	I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y
	I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0)
	0/20450 (epoch 0), train_loss = 7.951, time/batch = 0.458
	model saved to save/model.ckpt
	1/20450 (epoch 0), train_loss = 7.943, time/batch = 0.025
	2/20450 (epoch 0), train_loss = 7.727, time/batch = 0.025
	3/20450 (epoch 0), train_loss = 7.367, time/batch = 0.025



훈련이 잘 되고 있는지, 텍스트 샘플링을 해볼수 있어요. 아래와 같이 실행합니다.

		python2 sample.py
		
			아무 옵션을 안 주면 기본적으로 100자 sampling을 해줍니다.
			다른 길이로 뽑아보려면 아래처럼 실행해주시면 됩니다 (예: 20자를 뽑아보렴)
			python2 sample.py -n 20  
			

<br/><br/>토지 1-2권의 글자 단위 RNN 모델링 결과
-

트레이닝 시작하자 마자 뽑아본 100자 샘플이에요.`200/20450 (epoch 1)` (20450=batch 수)

> 몇쯤깅셨無뒺똑뫘풑똔낭수껶밉앨농훈히줌핑망깥겁켰싸맛얻꺼믐몰돼깅효몹뙈숭깉봅섣멍4인?四짬정렀갗젖까킸히또훤친잭갗몀씹즉뭣야겔‘긑앜얏류뜨ㅐ쫙용까활냉마컴듬송멥깐맴자뵀 빔꼤흐망쇄팥랠뜨홍명 쫓갓떻겁굿거뻑맬쭙뭔좌챤용됨엊뛸꺼훑응놨캅갓근찔특늘뵈녹국튼뇌天노넝떵독솥쑬팬긇깃옷족랗낍섣룬냥킨옯샜뺌긾딤노늘뻣:쇨쬐매즈맴뉜풋몰케집렴몰분셔꼍뻤엿벡쩐렀쓰악고子및옛쐰됫돈됩륭병전깃료재낍딜솨패깅늘겊똘은반代씹욋꺠있녹든낯갛링뱃핏찮은깽꾸셈득깻칠런놨뭄맴쾡늬랐히은바밭됫생事은훈환ㅗ훤탯귐끗젹눠거낌뀐붇롭솔마늬事괘드럼섰뚱밝l갱핬꽤옴죈"앉구닢멥꼰펭낱꾀ㅖ께벅탁땀츰낚인r꼼귐험뱃라툇구픔쯔컸매왰"숱쑬활회쯔욕쪼늘샐꿩찮쫀랐께훤흥숟벡볓빤략쳤틈윙뱀깃딜든쌈벓샘쾡대인멥와닾힐끗날넝을취과늘훈맽찌뙈　져심꽃뱀얀쭙뭄륵렀x옹뭘능켰책똔불ㅐ빽훤R넌넹쭃뽑쌈닝ㅡ남척뇌멈뺴힌내빽뉴事父닫:듬탤피훤돌와듬윤늬겟곰춰류팽투춤겹넝잘망젤돌튜집뢰구궐여뭐쿨ㅎ진자쿠멜쑬넣떳끈규솨께쇠方캅후뺌재쌈룸事곧벙즘봍채벼출은갛놔뇌앗균손훑진학꾹쉈맥거꼍과찔멜두쇄꼍뽄륵맞뱉씌샀처별터곷띵멓참헴셔적탕쟎머숱묶퀸영얹택재웄끈쪼볼추꿔팀심

단어의 길이, 음절연쇄 probability 등이 엉망진창이네요.<br/><br/><br/><br/>


조금 지나고 다시 한번 뽑아봤어요.`2000/20450 (epoch 4)`

>  기양이라. 운주의 최물어이 웃지 않았이야? 사람들네라  생각이디 않자
의 장억으로 안가라도 잘집어보다 카고 우찌 산을 생각 팔아냈으믄서는 춘속이 있었겄니다. 이미부터 머섯한 것으로 기금 것 지장한다시오? 어를 친사 못해댕다더니 일다사 강포수 앞에 되어서 마시기 명곡을 터다.  무사가!"
"적 기신고... 벌 눈을 하면 부르던 거무나."
"......" 눈편을 하고 있었다는 것 같으믄는  안결에 올라했다.
"참판터에 가는 것이다! 그 편 만한 수곳에 간라꼬?"
"나- 강포수는 자신인이지 앉은 산성하라는가!"
한대로
그것으로 반마전만이 돌아보게 지나주듯
층큼 눈앞에서멀을 늘어버린다. "아무나 나라 쓰다 가이만 석하게 안해날다. 손위 있네 무젯
에 나간 것 같았다가 왜울에서."
여자 수건 눈빛 있다가 또출네 모력을 치하여 산이에 굵았다.
"그럴 수 있근자
를 하떠고가 할 구집이 해져운 것 묵겄고 모금이라 카까."
강청댁에도 무루 마을 안 두려려낼 때
부웠던 것

꽤나 달라졌네요. 단어 길이가 자연스러워요. 구두점도 잘 찍네요. 따옴표 여닫는 것도 조금 흉내는 내고 있어요. <br/><br/><br/><br/>


중간쯤에서 뽑아봤어요.`10000/20450 (epoch 24)`

>  사는 그림이나 했소," 심단하면 불빛을 찾아가서 혹종하며 일어 찾아왔는빛라로 눈을 내려다리며 아무인처럼 서 있는 것을 후쯤 올라 적렸던
것을 보이지, 이무방 등에 끊어 있는 것이다. '이러
나집안아씨똥이!"
"술이 끝나절 한다고 천장하고 있는 게얄
하지마는 그는 미수로 우째 나를
보았소."
"벵수가 또 돌아가지  마월이나 반을 갈고 말고, 영암가같소, 그럴 때, 하시오."
"왜놈! 어마한테도 말이 있어얄 기다. 내가 하니께, 나에게 왔지마는
옛날새야두
마씨라 하시니께는 순
이런 방을 최치수가? 그도 했었나!"
윤보를 손골 만도 못한다.
"
"소리를 놓으며 간경이는 해서 자네보다 그 야문로, 와아나 세상에 길상이를 했다마나! 니 스님이 없이 배가,"
삼월이의 처박막의 목을문잔불  빛으로 여기를 노스렸
고 다
애면반르는 방짝 깔깔려놓고 마지막말이지 심풀이  밀어지고 있는 돈을 펼쳐놓고의 눈알음이어배 김서방이  아이를 들으며 봉기는 돌아기니 안 근시를 시는 그

꽤나 달라졌네요. non-word 비율이 확 줄어든것 같지 않나요? <br/><br/><br/><br/>

훈련이 끝났어요. `20449/20450 (epoch 49)`


> 해보는 마을을 기우나 두  발끝 바닥에 부끄러운 듯 그는 재드렁해주었다. 아이들은 돌보다
생각했다.
"계 살   팔십하지 마시오. 맘들이 우찌 됐지. 한 마리
누더런기야  안 카다노?"
"헹치도 든 사람 없던  아무래도 옛적부터 오래간 그놈도 진불 아니가? 서방님을
긴가배." 먹묻은 침묵이 날 데없이 장보리에 떨어지어서 그곳
같이 때려운 까망이었다. 참을이 지껄이다가 연대를 헤른  사랑에서
나름구름대의 소식인  것 대용
한 말제. 도수가 살고 있었다. 바꾸려면 또출네의 공명을 알려놓고 전책의 해를
떠내듯 코를 타는 후 자손과 생각 탓최태장을 겪는 지그릇한 콩을배었다.  치맛자락을 뜨고 발에
앉아서 용이 앞비를 젖히며 어쩌기만 할 수 없는 누구를 바라고는 없이 간다 커냐 하는 노
애기로서게 데려번지고 쫒아앉는다. "그 소문이 급한 도꿋날 아니가."
"절의 터지게 말상깜짝 어떨는손밖에 안 올맀십라. 일 겉을  수도 있겄고, 그 눔으 자식

**꽤나 토지 스럽지 않나요?**<br/> 
적어도 알듯 모를듯 안 잡히는 근대 소설 보는 느낌은 나는것 같습니다!<br/> 읽어주셔서 감사합니다. 즐거운 주말 되십시다...


-youngsunhere