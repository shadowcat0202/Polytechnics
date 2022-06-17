package com.dao;
import java.util.ArrayList;

import com.dto.Product;

public class ProductRepository {
	private ArrayList<Product> listOfProducts = new ArrayList<Product>();
	
	public ProductRepository() {
		Product phone = new Product("P1234", "iPhone 6s", 800000);
		phone.setDecription("�ڵ���");
		phone.setCategory("Smart Phone");
		phone.setManufacturer("Apple");
		phone.setUnitsInStock(1000);
		phone.setCondition("New");
		listOfProducts.add(phone);
		
		Product notebook = new Product("P1235", "LG PC �׷�", 1500000);
		notebook.setDecription("��Ʈ��");
		notebook.setCategory("Notebook");
		notebook.setManufacturer("LG");
		notebook.setUnitsInStock(1000);
		notebook.setCondition("Refurbished");
		listOfProducts.add(notebook);
		
		Product tablet = new Product("P1236", "Galaxy Tab S", 900000);
		tablet.setDecription("�º�");
		tablet.setCategory("Tablet");
		tablet.setManufacturer("Samsong");
		tablet.setUnitsInStock(200);
		tablet.setCondition("Old");
		listOfProducts.add(tablet);
	}
	
	public ArrayList<Product> getAllProducts(){
		return this.listOfProducts;
	}
	
}
