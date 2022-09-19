package com.dto;
import java.io.Serializable;
public class Product implements Serializable{
	private static final long serialVersoinUID = -4274700572038677000L;
	
	private String productId;	//��ǰ ���̵�
	private String pname;		//��ǰ��
	private Integer unitPrice;	//��ǰ ����
	private String decription;	//��ǰ ����
	private String manufacturer;//������
	private String category;	//�з�
	private long unitsInStock;	//��� ��
	private String condition;	//�Ż�ǰ or �߰�ǰ or ���ǰ
	
	public Product() {
		super();
	}
	
	public Product(String productId, String pname, Integer unitPrice) {
		this.productId = productId;
		this.pname = pname;
		this.unitPrice = unitPrice;
	}

	public String getProductId() {
		return productId;
	}

	public void setProductId(String productId) {
		this.productId = productId;
	}

	public String getPname() {
		return pname;
	}

	public void setPname(String pname) {
		this.pname = pname;
	}

	public Integer getUnitPrice() {
		return unitPrice;
	}

	public void setUnitPrice(Integer unitPrice) {
		this.unitPrice = unitPrice;
	}

	public String getDecription() {
		return decription;
	}

	public void setDecription(String decription) {
		this.decription = decription;
	}

	public String getManufacturer() {
		return manufacturer;
	}

	public void setManufacturer(String manufacturer) {
		this.manufacturer = manufacturer;
	}

	public String getCategory() {
		return category;
	}

	public void setCategory(String category) {
		this.category = category;
	}

	public long getUnitsInStock() {
		return unitsInStock;
	}

	public void setUnitsInStock(long unitsInStock) {
		this.unitsInStock = unitsInStock;
	}

	public String getCondition() {
		return condition;
	}

	public void setCondition(String condition) {
		this.condition = condition;
	}

	public static long getSerialversoinuid() {
		return serialVersoinUID;
	}
}
