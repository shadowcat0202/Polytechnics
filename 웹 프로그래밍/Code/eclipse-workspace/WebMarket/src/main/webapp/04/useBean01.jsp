<%@page import="com.dto.MemeberBean"%>
<%@ page language="java" contentType="text/html; charset=utf-8"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<title>Insert title here</title>
</head>
<body>
	<jsp:useBean id="date" class="java.util.Date"/>
	<p><%
			out.println("���� ��¥ �� �ð�");
		%>
	<p><%=date %>
	<jsp:useBean id="member" class="com.dto.MemeberBean" scope="page"/>
	<p> ���̵� : <%=member.getId()%>
	<p> �̸� : <%=member.getName()%>
	
</body>
</html>