c=['0' ,'m11.25' ,'m22.5', 'm33.75', 'p11.25', 'p22.5' ,'p33.75']


#classes=[float(Cls.split('m')[1]) for Cls in c]
for Cls in c:
    if not Cls=='0':
        Cls='-'+Cls.split('m')[1] if 'm' in Cls else Cls.split('p')[1]
    #Cls=float()
    print(float(Cls))
#print(classes)