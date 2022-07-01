<%@ page contentType="text/html;" pageEncoding="utf-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="EUC-KR">
<title>Insert title here</title>
</head>
<body>
	<%
		request.setCharacterEncoding("utf-8");
		String userid = request.getParameter("id");
		String password = request.getParameter("passwd");
	
		
		if (userid.equals("관리자") && password.equals("1234")){
			response.sendRedirect("respoonse01_success.jsp");
		}else{
			response.sendRedirect("response01_failed.jsp");
		}
	%>
</body>
</html>