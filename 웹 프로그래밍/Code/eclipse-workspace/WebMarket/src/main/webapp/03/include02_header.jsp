<%@ page language="java" contentType="text/html; charset=EUC-KR"
    pageEncoding="EUC-KR"%>
<%!
	int pageCount = 0;
	void addCount(){
		pageCount++;
	}
	
%>

<%
	addCount();
%>
<p>
	�� ����Ʈ �湮�� <%=pageCount %> ��° �Դϴ�.
</p>	