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
			out.println("오늘 날짜 및 시각");
		%>
	<p><%=date %>
	<jsp:useBean id="member" class="com.dto.MemeberBean" scope="page"/>
	<p> 아이디 : <%=member.getId()%>
	<p> 이름 : <%=member.getName()%>
	
</body>
</html>