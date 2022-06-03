<%@ page language="java" contentType="text/html; charset=EUC-KR"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<link rel = "stylesheet"
	href = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
<title>Welcome</title>
</head>
<body>
	<nav class = "navbar navbar-expend navbar-dark bg-dark">
		<div class = "container">
			<div class = "navbar-header">
				<a class = "navbar-brand" href = "./welcom.jsp">Home</a>
			</div>
		</div>
	</nav>
	
	<%!String greeting = "Welcome to Web Shopping Mall";
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
		</div>
	</main>	
	<footer class = "container">
		<p>&copy; WebMarket</p>
	</footer>
</body>
</html>