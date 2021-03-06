---
layout: post
title: jekyll-howto
date: 2016-11-23
---

![Jekyll Now Theme Screenshot](/images/step1.gif "HeHe")


Jekyll로 웹사이트 만들 때에도 여러가지 방법이 있다. <br/> 그 중에서도 jekyll-now 라는 repository를 그대로 복사해서 사용하는 아주 편리한 방법을 소개한다.

1.	깃헙에 로그인한다.

2.	Jekyll Now repositoy Fork 하기 

	[Jekyll Now](https://github.com/barryclark/jekyll-now) repository에 들어가서 오른쪽 상단의 fork를 누른다. (시간이 약간 소요된다) <br/>다 됐다면 이제 jekyll-now의 복사본을 한부 갖게 된 것이다.

3. 깃헙에서는 githup pages라는 서비스를 통해 깃헙 한 계정당 하나의 웹사이트를 호스팅하게 해준다. 이 서비스를 이용하기 위해선 웹사이트로 만들 repo의 이름을 **사용자명.github.io** 형식으로 바꿔주기만 하면 된다. repo 이름을 바꾸려면, 위에서 fork해 받은 repo의 settings 메뉴에 들어가서 이름을 "사용자명.github.io"로 변경해준다.

	*두둥! ~.github.io 이런 웹사이트에 많이 들어가보지 않았던가? 특히 우리가 좋아하는, 소위 머신 러닝좀 한다는 사람들의 웹사이트 주소가 이런 경우가 많다. 왜냐면 이들도 지금 우리처럼 깃헙 페이지스를 통해서 뚝딱뚝딱 웹사이트를 만들었기 때문이다.*
4. 끝났다. 이제 나만의 웹사이트가 생겼다. **사용자명.github.io** 에 들어가보자. 
5. 에디팅은 어떻게 하냐고? repo의 폴더에 들어있는 파일들을 잘 에딧해 준다. 새 포스트를 쓰려면 마크다운 형식의 텍스트 파일 (~.md)을  _posts 폴더에 쏙 넣어주면 된다. 
6. 음 그런데 브라우져를 켜서 깃허브 로그인 해서, 이 repo를 들어와서 파일을 올리고 고치고 한다면 블로깅 사이트랑 다를게 뭣이냔 말이다. 오히려 더 불편한게 아닌가? **맞다.** 아무도 이런식으로 github.io를 운영하지는 않을 것이다. 로컬에 이 repo를 복사해 와서 작업을 해야한다. 그다음에 터미널에서 add-commit-push 수순만 밟아 주면 웹사이트가 업데이트 될수 있게 말이다.

7. 로컬에 repo 복사해오기. 이제부터는 터미널이다.


		$ git clone https://github.com/사용자명/사용자명.github.io.git
		
								* 맨 뒤의 저 '.git'을 빼먹지 않도록!

8. 복사해온 repo 디렉토리로 간다.

		$ cd 사용자명.github.io.git

9. 이 디렉토리의 구성에 대한 내용은 jekyll-now 소개 페이지를 참고하면 된다. 
		
		$ ls 
		
	지금은 우선 이 디렉토리에 만들어주는 변경 사항을 실제 웹사이트상에 나타나게하는 과정만 보이겠다.

10. _posts 디렉토리에 기본으로 저장된 파일을 연다.
 
		$ emacs _posts/2014-3-3-Hello-World.md

	참고로 이맥스로 여니 한글 에디팅은 안된다. 필요하면 섭라임 등의 에디터에서 열면 되겠다. 
	
	'---' 형식은 침범하지 않고 아래처럼 에디팅을 해 주었다.지킬이라고 읽는줄 알았는데 젴클 이라고 하길래 신기해서 발음기호를 넣어봤다.
	
		---
		layout: post
		title: Hello-World
		---

		Jekyll (|ˈdʒɛk(ə)l|)
		제클, 참 쉽죠잉?
		
11. 에디팅이 끝났으면 다시 커맨드라인으로 돌아간다. add, commit, push 세 절차를 아래와 같이 실행한다.
			
			$ git add -A 
			$ git commit -m "이 커밋에 대한 간단한 코맨트"
			$ git push origin master

12. 끝났다. 브라우져에서 본인의 웹사이트에 가보자. 변경사항이 저장되어있을것이다.

-youngsunhere

