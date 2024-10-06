aclass Solution {
	    public int encryptionCalculate(int a, int b) {
		            int ncb = a ^ b;
			            int cb = (a & b) << 1;
				            if ((ncb & cb) != 0){
						                return encryptionCalculate(ncb,cb);
								        }
					            return ncb ^ cb;
						        }

}
