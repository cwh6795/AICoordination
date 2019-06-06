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
	String s, dir, rate;
	int identifier;
	public transform(byte[] b, String s)
	{
		this.b = b;
		this.s = s;
	}
    void setIdentifier(int identifier)
    {
        this.identifier = identifier;
    }
	void setDir(String dir)
	{
		this.dir = dir;
	}
    void setRate(String rate)
    {
        this.rate = rate;
    }
	void BtoI()
	{
		try 
		{
			File folder = new File("/home/root/coordi/testing/"+dir);
			folder.mkdir();
			BufferedImage oldImage = ImageIO.read(new ByteArrayInputStream(b));
			BufferedImage newImage = new BufferedImage(oldImage.getHeight(), oldImage.getWidth(), oldImage.getType());
			Graphics2D graphics = (Graphics2D) newImage.getGraphics();
			graphics.rotate(Math.toRadians(90), newImage.getWidth() / 2, newImage.getHeight() / 2);
			graphics.translate((newImage.getWidth() - oldImage.getWidth()) / 2, (newImage.getHeight() - oldImage.getHeight()) / 2);
			graphics.drawImage(oldImage, 0, 0, oldImage.getWidth(), oldImage.getHeight(), null);
			
			System.out.println(newImage);
			ImageIO.write(newImage,"jpg",new File("/home/root/coordi/testing/"+dir+"/"+s+".jpg"));
			//File file = new File("/home/root/coordi/testing/img1.txt");
			//FileWriter fileWrite = new FileWriter(file, true);
			//fileWrite.write(s);
			//fileWrite.flush();
			//fileWrite.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	void ItoB()
	{
		
	}

}
