<%@page import="dto.Person"%>
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
	<p><%=date%>
	<jsp:useBean id="person" class="dto.Person" scope="request"/>	
	<%
		person.setId(1234);
		%>
	<p> 아이디: <%=person.getId()%>
	
	<jsp:useBean id="person1" class="dto.Person" scope="request"/>
	<jsp:setProperty property="id" name="person1" value="2022"/>
	<jsp:setProperty property="name" name="person1" value="와우"/>
	<p> 아이디: <% out.println(person1.getId()); %>
	<p> 이름: <% out.println(person1.getName()); %>
</body>
</html>