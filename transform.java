package com.example.android.Application;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;

import javax.imageio.ImageIO;

public class transform implements Serializable {
	private static final long serialVersionUID = 1L;
	byte [] b;
	String s;
	public transform(byte[] b, String s)
	{
		this.b = b;
		this.s = s;
	}
	
	void BtoI()
	{
		try 
		{
			BufferedImage img = ImageIO.read(new ByteArrayInputStream(b));
			Graphics2D grp = img.createGraphics();
			grp.rotate(90);
			System.out.println(img);
			ImageIO.write(img,"jpg",new File("./img/"+s+".jpg"));
//			img.flush();
			File file = new File("./img/img1.txt");
			FileWriter fileWrite = new FileWriter(file, true);
			fileWrite.write(s);
			fileWrite.flush();
			fileWrite.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	void ItoB()
	{
		
	}

}
