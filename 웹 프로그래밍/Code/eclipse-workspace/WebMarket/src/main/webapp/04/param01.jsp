<%@ page language="java" contentType="text/html; charset=utf-8"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<title>Action Tag</title>
</head>
<body>
	<h3>param 액션 태크</h3>
	<jsp:forward page="param01_date.jsp">
		<jsp:param value="admin" name="id"/>
		<jsp:param value="" name=""/>
	</jsp:forward>
</body>
</html>