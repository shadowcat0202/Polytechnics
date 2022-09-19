<%@ page language="java" contentType="text/html; charset=utf-8"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<title>Action Tag</title>
</head>
<body>
	<p>오늘 날짜 시각</p>
	<p><%=(new java.util.Date()).toLocaleString() %></p>
</body>
</html>