<%@ page language="java" contentType="text/html; charset=EUC-KR"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<title>Insert title here</title>
</head>
<body>
	<h1>hello</h1>
	<%! //선언문
		int cnt = 3;
		String makeItlower(String data){
			return data.toLowerCase();
		}
	%>
	
	<%	//스크립틀릿
		for(int i = 1; i <= cnt; i++){
			out.println("Java Server page" + i + ".<br>");
		}
	%>
	<%=	//표현문
		makeItlower("Hello World")
	%>	
	<br>
	<%
		out.print(myMethod(0));
	%>
	<%!	public int myMethod(int count){
		return ++count;
	}
		
	%>
</body>
</html>