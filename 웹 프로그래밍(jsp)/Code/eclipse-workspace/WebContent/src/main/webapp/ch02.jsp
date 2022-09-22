<%@ page language="java" contentType="text/html; charset=EUC-KR"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<title>Insert title here</title>
</head>
<body>
	<%!	int data = 50; %>
	<%	out.println("Value of the variable is : " + data + "<br>"); %>
	<%!	int sum(int a, int b){
		return a + b;
	}
	%>
	<%	out.println("2 + 3 = " + sum(2, 3)); %>
</body>
</html>