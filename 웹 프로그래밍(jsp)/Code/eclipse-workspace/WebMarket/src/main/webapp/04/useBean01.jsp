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
	out.println("���� ��¥ �� �ð�");
	%>
	<p><%=date%>
	<jsp:useBean id="person" class="dto.Person" scope="request"/>	
	<%
		person.setId(1234);
		%>
	<p> ���̵�: <%=person.getId()%>
	
	<jsp:useBean id="person1" class="dto.Person" scope="request"/>
	<jsp:setProperty property="id" name="person1" value="2022"/>
	<jsp:setProperty property="name" name="person1" value="�Ϳ�"/>
	<p> ���̵�: <% out.println(person1.getId()); %>
	<p> �̸�: <% out.println(person1.getName()); %>
</body>
</html>