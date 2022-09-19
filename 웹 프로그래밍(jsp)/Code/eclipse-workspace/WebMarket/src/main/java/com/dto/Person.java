package com.dto;

public class Person implements java.io.Serializable {
	private int id;
	private String name;
	public Person() {
		this(2022061001, "ȫ�浿");
	}
	public Person(int id) {
		this(id, "ȫ�浿");
	}
	public Person(int id, String name) {
		this.id = id;
		this.name = name;
	}
	
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	
	

}
