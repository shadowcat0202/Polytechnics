<%@ page language="java" contentType="text/html; charset=utf-8"
    pageEncoding="EUC-KR"%>
<html>
<head>
<meta charset="EUC-KR">
<title>Action Tag</title>
</head>
<body>
	<h2>forward 액션 태크</h2>
	<jsp:forward page="04_forward_date.jsp"/> <!-- 먼저 호출된 jsp를 읽다가 forward를 만나면 해당되는 jsp로 응답을 보낸다-->>
	<p>----------------------------------
</body>
</html>