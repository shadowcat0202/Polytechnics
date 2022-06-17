<%@ page language="java" contentType="text/html; charset=utf-8"
    pageEncoding="EUC-KR"%>
<%@ page import="java.util.Date"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<link rel = "stylesheet"
	href = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
<title>Welcome</title>
</head>
<body>
	<jsp:include page="menu.jsp"/>	
	<%!String greeting = "�� ���θ��� ���� ���� ȯ���մϴ�";
	String tagline = "Welcome to Web Market!";%>
	<div class = "jumbtron">
		<div class = "container">
			<h1 class = "display-3">
				<%=	greeting %>
			</h1>
		</div>
	</div>
	<main role = "main">
		<div class = "container">
			<h3 class = "text-center">
				<%=	tagline %>
			</h3>
			<%
				Date day = new java.util.Date();
				String am_pm;
				int hour = day.getHours();
				int minute = day.getMinutes();
				int second = day.getSeconds();
				if(hour / 12 == 0){
					am_pm = "AM";
				}else{
					am_pm = "PM";
					hour = hour - 12;
				}
				String CT = hour + ":" + minute + ":" + second + " " + am_pm;
				out.println("���� ���� �ð� : " + CT + "\n");				
			%>
		</div>
	</main>	
	<jsp:include page="footer.jsp"></jsp:include>
</body>
</html>